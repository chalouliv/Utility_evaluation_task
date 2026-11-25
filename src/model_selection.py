from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from src import load_real

NAME = "model_selection"
REGRESSION_FAMILIES = ["linReg", "linRidge", "linLasso"]
CLASSIFICATION_FAMILIES = ["logL1", "logL2"]

SCORING_REGRESSION = {
    "neg_mae": "neg_mean_absolute_error",
    "neg_rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
SCORING_BINARY_CLASSIFICATION = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "neg_log_loss": "neg_log_loss"
}
SCORING_MULTICLASS_CLASSIFICATION = {
    "f1": "TBA",
    "roc_auc": "TBA",
    "neg_log_loss" : "TBA"
}
# NEGATED = {"mae", "rmse", "log_loss"} ## sklearn's "neg_*" flip them back, not using 

DEFAULT_SELECTION = {"regression": "neg_mae", "classification": "roc_auc"}

def alpha_grid(cfg) -> np.ndarray:
    grid = cfg["models"]["alpha_grid"]
    return np.logspace(grid["start"], grid["stop"], grid["num"])

def c_grid(cfg) -> np.ndarray:
    grid = cfg["models"]["c_grid"]
    return np.logspace(grid["start"], grid["stop"], grid["num"])

def make_preprocess(cfg) -> ColumnTransformer:
    ## One-hot encode categorical columns (type: categorical in config), scale all other columns
    columns = cfg["data"]["columns"]
    ## Isolate categorical columns
    cats = {c for c, spec in columns.items() if spec.get("type") == "categorical"}
    ## Returns instructions for sklearn
    return ColumnTransformer(
        [
            ("num", StandardScaler(),
             lambda X: [c for c in X.columns if c not in cats]),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
             lambda X: [c for c in X.columns if c in cats]),
        ],
        verbose_feature_names_out=False ## keeps og column names
    )

def pipeline(cfg, family: str) -> Pipeline:
    ## Scale numeric columns, one-hot encode categoricals, then fit the model.
    ## Ridge/Lasso tune alpha and logistic models tune C by inner CV.
    n_folds=cfg["cv"]["n_folds"]
    seed = cfg["run"]["seed"]
    if family == "linReg":
        model = LinearRegression()
    elif family == "linRidge":
        model = RidgeCV(
            alphas=alpha_grid(cfg),
            cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed),
            scoring="neg_mean_squared_error",
        )
    elif family == "linLasso":
        model = LassoCV(
            alphas=alpha_grid(cfg),
            cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed),
            random_state=seed,
            max_iter=5000,
        )
    elif family == "logL1":
        model = LogisticRegressionCV(
            Cs=c_grid(cfg), 
            penalty="l1",
            solver="saga", 
            cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed), 
            scoring = "roc_auc", 
            random_state=seed,
            max_iter=5000
        )
    elif family == "logL2": 
        model = LogisticRegressionCV(
            Cs=c_grid(cfg), 
            penalty="l2",
            solver="lbfgs", 
            cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed),
            scoring="roc_auc",
            random_state=seed,
            max_iter=5000
        )
    else:
        raise ValueError(f"Unknown family {family!r}")
    return Pipeline([("preprocess", make_preprocess(cfg)), ("model", model)])

def chosen_alpha(pipe: Pipeline) -> float:
    model = pipe.named_steps["model"]
    return float(model.alpha_) if hasattr(model, "alpha_") else np.nan

def chosen_C(pipe: Pipeline) -> float:
    model = pipe.named_steps["model"]
    return float(np.ravel(model.C_)[0]) if hasattr(model, "C_") else np.nan

def run(cfg, dpath) -> None:
    ## Define output paths and skip if already done
    out_dir = dpath / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / "model_selection.csv"
    if fpath.exists():
        print(f"[{NAME}] {fpath} exists, skipping. Delete it to rerun.")
        return
    
    ## Load real and get list of targets 
    real = load_real.load(cfg)
    targets = cfg["data"]["target"]
    specs = {name: spec["type"] for name, spec in targets.items()} ## compress targets list

    ## Rows collects results of each model 
    rows = []
    n_folds = cfg["cv"]["n_folds"]
    seed = cfg["run"]["seed"]

    ## Loop over each target and isolate appropriate X and y
    for target, task in specs.items():
        data = load_real.get_xy(real, "train", target, task)
        X = data["X"]
        y = data["y"] #already encoded if needed 

        ## For each target picks families (models to compare), scoring (which metrics to compute), cv (outer fold spliter)
        if task == "regression":
            families, scoring, cv = REGRESSION_FAMILIES, SCORING_REGRESSION, KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        elif task == "classification":
            families, scoring, cv = CLASSIFICATION_FAMILIES, SCORING_BINARY_CLASSIFICATION, StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            raise ValueError(f"Unknown type {task} for target {target}") ## Error if task is unrecognized 
        ## print(f"[{NAME}] {target} ({task}): X={X.shape}, families={families}")

        for family in families:
            ## create new unfited model with innerCV for hyperparameter tuning
            pipe = pipeline(cfg, family)

            ## run outer loop of cross validation 
            result = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_estimator=True)
            for fold, est in enumerate(result["estimator"]):
                row = {
                    "target": target,
                    "task": task,
                    "family": family,
                    "fold": fold,
                    "inner_alpha": chosen_alpha(est),
                    "inner_C": chosen_C(est),
                }
                for key in scoring:
                    score = result[f"test_{key}"][fold]
                    row[key] = score
                rows.append(row)

    ## Order results and find 'best' model 
    df = pd.DataFrame(rows) ## one row per target, famil, and fold
    df["chosen"] = False
    df["selection_metric"] = ""
    for target, grp in df.groupby("target", sort=False):
        ## Sort results of df by target
        task = grp["task"].iloc[0] # task is same across all of the same target 
        metric = DEFAULT_SELECTION[task]
        ## Take means across folds for each family and isolate family with highest mean 
        means = grp.groupby("family")[metric].mean() 
        best = means.idxmax()
        ## Summarize results 
        df.loc[df["target"] == target, "selection_metric"] = metric 
        df.loc[(df["target"] == target) & (df["family"] == best), "chosen"] = True
        summary = grp.groupby("family")[metric].agg(["mean", "std"]).round(4)
        print(f"[{NAME}] {target}: outer-fold CV on real train (chosen by {metric}: {best})\n{summary}")

    ## Save results to model_selection.csv
    df["cv_random_state"] = cfg["run"]["seed"]
    df["model_random_state"] = cfg["run"]["seed"]
    df.to_csv(fpath, index=False)
    return print(f"Done {NAME}.")
from pathlib import Path
import numpy as np 
import pandas as pd
from functools import partial
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import load_real
 
NAME = "model_selection"
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
LOWER_IS_BETTER = {"mae": True, "rmse": True, "r2": False}
 
def alpha_grid(config: dict) -> np.ndarray:
    grid = config["models"]["alpha_grid"]
    return np.logspace(grid["start"], grid["stop"], grid["num"])
 
def make_cv(config: dict) -> KFold:
    return KFold(
        n_splits = config["cv"]["n_folds"],
        shuffle = config["cv"]["shuffle"],
        random_state = config["run"]["seed"] if config["cv"]["shuffle"] else None,
    )
 
def categorical_columns(config: dict) -> set[str]:
    columns = config["data"]["columns"] 
    return {c for c, spec in columns.items() if (spec or {}).get("type") == "categorical"}

def _select(X: pd.DataFrame, cats: frozenset, want_cat: bool) -> list[str]:
    return [c for c in X.columns if (c in cats) == want_cat]

def make_preprocess(config: dict) -> ColumnTransformer:
    ## One-hot encode categorical columns and scale numerical columns 
    cats = frozenset(categorical_columns(config))
    return ColumnTransformer(
        [
            ("num", StandardScaler(), partial(_select, cats=cats, want_cat=False)),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
             partial(_select, cats=cats, want_cat=True)),
        ],
        verbose_feature_names_out=False, 
    )

def build_pipeline(config: dict, family: str) -> Pipeline:
    ## Inner CV (ridge and lasso only)
    ## Scale numeric columns, one-hot encode categoricals, then return three models pipelines.
    ## Ridge and Lasso determine an alpha by Inner CV.
    ## Vanilla linear
    if family == "ols": 
        model = LinearRegression()
    ## Linear with ridge regulation
    elif family == "ridge": 
        model = RidgeCV(
            alphas=alpha_grid(config), 
            cv=make_cv(config), 
            scoring="neg_mean_squared_error")  # Score by MSE so ridge uses the same selection rule as lasso.
    ## Linear with lasso regulation
    elif family == "lasso":
        model = LassoCV(
            alphas=alpha_grid(config),
            cv=make_cv(config),
            random_state=config["run"]["seed"],
            max_iter=100_000,
        )
    else:
        raise ValueError(f"Unknown family {family!r}")
    return Pipeline([("preprocess", make_preprocess(config)), ("model", model)])
 
def chosen_alpha(pipe: Pipeline) -> float:
    ## return best alpha for each of the fitted model pipelines 
    model = pipe.named_steps["model"]
    return float(model.alpha_) if hasattr(model, "alpha_") else np.nan

def run(config: dict, dpath: Path) -> None:
    ## Run nested CV to test and select alphas for Ridge and Lasso
    ## Inner CV tunes alpha on each outer training split (ridge and lasso only)
    ## Outer CV scores model on each outer testing split
    ## Family with lowest mean MSE across outer folds wins (marked chosen=True)

    ## Create paths for outputs and check if they already exist 
    out_dir = dpath / NAME
    print(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = dpath / NAME / "model_selection.csv"
    if fpath.exists():
        print(f"[{NAME}] {fpath} exists, skipping. Delete it to rerun.")
        return
    
    metric = config["metrics"]["selection_metric"]
    n_folds = config["cv"]["n_folds"]
    real = load_real.load(config)
    outer = make_cv(config)
    rows = []
    for family in config["models"]["families"]:
        pipe = build_pipeline(config, family)
        res = cross_validate(
            pipe, real["X_train"], real["y_train"], cv=outer, scoring=SCORING, return_estimator=True
        )
        for fold in range(n_folds):
            alpha = chosen_alpha(res["estimator"][fold])
            rows.append({
                "family": family,
                "fold": fold,
                "mae": -res["test_mae"][fold],
                "rmse": -res["test_rmse"][fold],
                "r2": res["test_r2"][fold],
                "inner_alpha": alpha,
            })
    df = pd.DataFrame(rows)
    means = df.groupby("family")[metric].mean()
    best = means.idxmin() if LOWER_IS_BETTER[metric] else means.idxmax()
    df["chosen"] = df["family"] == best
    df["cv_random_state"] = config["run"]["seed"]
    df["model_random_state"] = config["run"]["seed"]
    df.to_csv(fpath, index=False)
    summary = df.groupby("family")[["mae", "rmse", "r2"]].agg(["mean", "std"]).round(4)
    print(f"[{NAME}] outer-fold CV on real_train:\n{summary}")
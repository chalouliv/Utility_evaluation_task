from pathlib import Path
from random import shuffle
import numpy as np
import pandas as pd
from sklearn.metrics import ( f1_score, log_loss,
    mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error, balanced_accuracy_score,
)
from src import load_real, load_split_synthetic, model_selection

NAME = "evaluate_utility"

WINNERS = {"AGE": "linReg", "PTGENDER": "logL1"}  # from model_selection.csv, change for each task (not ideal)

'''
def score(y_true, y_pred, task) -> dict:
    ## regression scoring
    if task == "regression": 
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
    else:  #task == "classification"" 
        return {
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred),
            "neg_log_loss": log_loss(y_true, y_pred)
        }
'''

def predict_and_score(task: str, pipe, X, y):
    ## Returns (scores, predictions): predicted values for regression, P(positive class) for classification.
    ## Predictions are kept out of the scores dict so raw_results.csv stays one number per cell.
    if task == "regression" :
        y_pred = pipe.predict(X)
        return {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": root_mean_squared_error(y, y_pred),
            "r2": r2_score(y, y_pred),
        }, y_pred
    else: #task == "classification"
        y_proba = pipe.predict_proba(X)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        return {
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba)
        }, y_proba
    

def fit(cfg, family: str, X, y, features) -> dict:
    ## Fit a fresh pipeline of the winning family
    pipe = model_selection.pipeline(cfg, family)
    pipe.fit(X, y)
    C = model_selection.chosen_C(pipe)

    ## collect coefficients for later (need to add)
    model = pipe[-1]
    names = pipe[:-1].get_feature_names_out(features)
    coef = pd.Series(np.ravel(model.coef_), index=names)
    intercept = float(np.ravel(model.intercept_)[0])
    return {"pipe": pipe, "coef": coef, "intercept": intercept, "C": C}


def summarise(raw: pd.DataFrame, eps_order: list) -> pd.DataFrame:
    ## WRITTEN WITH CLAUDE ##

    ## Group each run and calculate mean and std
    grouped = raw.groupby(["target", "comparison", "eps"], dropna=False)
    metrics = [m for m in ["mae", "rmse", "r2", "f1", "roc_auc", "balanced_accuracy"] if m in raw]
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["n_seeds"] = grouped.size()
    summary = summary.reset_index()

    ## Create order for summary
    comp_order = {c: i for i, c in enumerate(["TRTR", "TSTR", "TSTS", "TRTS"])}
    e_order = {e: i for i, e in enumerate(eps_order)}
    t_order = {t: i for i, t in enumerate(raw["target"].unique())}
    summary["_t"] = summary["target"].map(t_order)
    summary["_c"] = summary["comparison"].map(comp_order)
    summary["_e"] = summary["eps"].map(e_order).fillna(-1)  # TRTR (no ε) sorts first

    return summary.sort_values(["_t", "_c", "_e"]).drop(columns=["_t", "_c", "_e"]).reset_index(drop=True)

def run(cfg, dpath=None) -> None:
    ## Create paths for outputs and check if they already exist
    out_dir = dpath / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"
    coefs_path = out_dir / "coefficients.csv"
    preds_path = out_dir / "predictions.csv"
    if preds_path.exists():
        print(f"[{NAME}] {preds_path} exists, skipping. Delete it to rerun.")
        return
    if coefs_path.exists():
        print(f"[{NAME}] {coefs_path} exists, skipping. Delete it to rerun.")
        return
    if raw_results_path.exists():
        print(f"[{NAME}] {raw_results_path} exists, skipping. Delete it to rerun.")
        return
    if summary_path.exists():
        print(f"[{NAME}] {summary_path} exists, skipping. Delete it to rerun.")
        return

    ## Load real data
    real = load_real.load(cfg)
    
    ## Gather synthetic data
    pairs = load_split_synthetic.list_pairs(cfg, dpath)

    ## Targets and their task type, e.g. {"AGE": "regression", "PTGENDER": "classification"}
    targets = cfg["data"]["target"]
    specs = {name: spec["type"] for name, spec in targets.items()} ## compress targets list

    rows = []
    coefs = [] 
    c = []

    ## Record coefficients + intercept of a fitted model
    def add_coefs(target, comparison, train_set, eps, seed, fit_result):
        for feature, value in fit_result["coef"].items():
            coefs.append({
                "target": target,
                "comparison": comparison,
                "train_set": train_set,
                "eps": eps,
                "seed": seed,
                "feature": feature,
                "coef": value,
            })
        coefs.append({
            "target": target,
            "comparison": comparison,
            "train_set": train_set,
            "eps": eps,
            "seed": seed,
            "feature": "(intercept)",
            "coef": fit_result["intercept"],
        })

    ## Easier for collection
    def add_row(target, comparison, train_set, test_set, eps, seed, n_train, c, scores):
        rows.append({
            "target": target,
            "comparison": comparison,
            "train_set": train_set,
            "test_set": test_set,
            "eps": eps,
            "seed": seed,
            "n_train": n_train,
            "C": c,
            **scores,
        })

    ## Raw predictions per test row for every run (all targets), saved to predictions.csv
    preds = []

    ## Score a fitted pipeline on (X, y), add the row, and keep the per-row predictions
    def score_and_record(target, task, comparison, train_set, test_set, eps, seed, n_train, c, pipe, X, y):
        scores, y_pred = predict_and_score(task, pipe, X, y)
        add_row(target, comparison, train_set, test_set, eps, seed, n_train, c, scores)
        classes = getattr(pipe, "classes_", None) if task == "classification" else None
        preds.append(pd.DataFrame({
            "target": target,
            "task": task,
            "comparison": comparison,
            "train_set": train_set,
            "test_set": test_set,
            "eps": eps,
            "seed": seed,
            "y_true": np.asarray(y),
            "y_pred": np.asarray(y_pred),  # regression: predicted value; classification: P(positive class)
            "positive_class": classes[1] if classes is not None else None,
        }))

    ## Fit and predict for TRTR: train real, test real (the baseline), per target.
    real_fits = {}
    real_xy = {}

    ## Null model results for comparision (train target shuffled)
    null_fits = {}

    for target, task in specs.items():
        ## Isolate X and y for each target
        train, test = load_real.get_xy(real, "train", target, task), load_real.get_xy(real, "test", target, task)
        X_tr, y_tr = train["X"], train["y"]
        X_te, y_te = test["X"], test["y"]
        real_xy[target] = (X_tr, y_tr, X_te, y_te)

        ## get feature names 
        features = X_tr.columns.tolist()

        # Fit fresh model of winning type for each task
        real_fits[target] = fit(cfg, WINNERS[target], X_tr, y_tr, features)

        ## Suffle y for each target in train 
        y_tr_shuffled = y_tr.sample(frac=1).to_numpy()

        # Fit null model of winning type for each task 
        null_fits[target] = fit(cfg, WINNERS[target], X_tr, y_tr_shuffled, features)

        ## Record coefficients for real and null fits
        add_coefs(target, "TRTR", "real", None, None, real_fits[target])
        add_coefs(target, "NULL", "real_suffled", None, None, null_fits[target])

        # Add results of TRTR to rows
        score_and_record(target, task, "TRTR", "real", "real", None, None, len(X_tr), real_fits[target]["C"],
                         real_fits[target]["pipe"], X_te, y_te)

        # Add results of NULL to rows
        score_and_record(target, task, "NULL", "real_suffled", "real", None, None, len(X_tr), null_fits[target]["C"],
                         null_fits[target]["pipe"], X_te, y_te)

    ## Iterate through synthetic datasets 
    for _, pair in pairs.iterrows():

        ## Load synthetic datasets for each epsilon and seed
        eps, seed = pair["eps"], pair["seed"]
        syn = {
            "train": pd.read_csv(pair["train"]),
            "test": pd.read_csv(pair["test"])
        }

        ## For each target 
        for target, task in specs.items():
            
            ## Isolate target from real
            X_real_tr, y_real_tr, X_real_te, y_real_te  = real_xy[target]

            features = X_real_tr.columns.tolist()
            
            ## Isolate target from synthetic
            syn_tr, syn_te = load_real.get_xy(syn, "train", target, task), load_real.get_xy(syn, "test", target, task)
            X_syn_tr, y_syn_tr = syn_tr["X"], syn_tr["y"]
            X_syn_te, y_syn_te = syn_te["X"], syn_te["y"]

            # Find model fited on real 
            real_pipe = real_fits[target]["pipe"]

            ## TRTS: train real, test synthetic
            score_and_record(target, task, "TRTS", "real", "synthetic", eps, seed, len(X_real_tr), real_fits[target]["C"],
                             real_pipe, X_syn_te, y_syn_te)

            ## TSTR: train synthetic, test real
            syn_fit = fit(cfg, WINNERS[target], X_syn_tr, y_syn_tr, features)
            ## Record coefficients of the synthetic fit (shared by TSTR and TSTS)
            add_coefs(target, "TSTR", "synthetic", eps, seed, syn_fit)
            score_and_record(target, task, "TSTR", "synthetic", "real", eps, seed, len(X_syn_tr), syn_fit["C"],
                             syn_fit["pipe"], X_real_te, y_real_te)

            ## TSTS: train synthetic, test synthetic (test from same dataset as train)
            score_and_record(target, task, "TSTS", "synthetic", "synthetic", eps, seed, len(X_syn_tr), syn_fit["C"],
                             syn_fit["pipe"], X_syn_te, y_syn_te)

    # Convert to dataframe and order by target, comparison metric, and epsilon
    raw = pd.DataFrame(rows)
    coef_df = pd.DataFrame(coefs)
    eps_order = list(pd.unique(pairs["eps"])) if len(pairs) else []
    summary = summarise(raw, eps_order)

    # Save
    raw.to_csv(raw_results_path, index=False)
    summary.to_csv(summary_path, index=False)
    coef_df.to_csv(coefs_path, index=False)
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(preds_path, index=False)
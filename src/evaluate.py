from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from src import load_real, load_split_synthetic, model_selection

NAME = "evaluate"

WINNER = "ols"

def score(y_true, y_pred) -> dict:
    ## easy scoring function
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

def fit(cfg, X, y, features: list[str]) -> dict:
    ## Fit a fresh OLS pipeline  
    pipe = model_selection.build_pipeline(cfg, WINNER)
    pipe.fit(X, y)

    ## collect coefficients for later (need to add)
    model = pipe[-1]
    names = pipe[:-1].get_feature_names_out(features)
    coef = pd.Series(np.ravel(model.coef_), index=names)
    intercept = float(np.ravel(model.intercept_)[0])
    return {"pipe": pipe, "coef": coef, "intercept": intercept}



def summarise(raw: pd.DataFrame, eps_order: list) -> pd.DataFrame:
    ## WRITTEN WITH CLAUDE ##
    metric_cols = [ "mae", "rmse", "r2"]

    ## Group each run and calculate mean and std
    grouped = raw.groupby(["comparison", "eps"])
    summary = grouped[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["n_seeds"] = grouped.size()
    summary = summary.reset_index()

    ## Create order for summary 
    comp_order = {c: i for i, c in enumerate(["TRTR", "TSTR", "TSTS", "TRTS"])}
    e_order = {e: i for i, e in enumerate(eps_order)}
    summary["_c"] = summary["comparison"].map(comp_order)
    summary["_e"] = summary["eps"].map(e_order).fillna(-1)  # TRTR (no ε) sorts first

    return summary.sort_values(["_c", "_e"]).drop(columns=["_c", "_e"]).reset_index(drop=True)


def run(cfg, dpath=None) -> None:

    ## Create paths for outputs and check if they already exist 
    out_dir = dpath / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"
    if raw_results_path.exists():
        print(f"[{NAME}] {raw_results_path} exists, skipping. Delete it to rerun.")
        return
    if summary_path.exists():
        print(f"[{NAME}] {summary_path} exists, skipping. Delete it to rerun.")
        return

    ## Load real data
    real = load_real.load(cfg)
    features = real["features"]

    ## Gather synthetic data
    pairs = load_split_synthetic.list_pairs(cfg, dpath)

    ## Fit and predict for TRTR: train real, test real (the baseline).
    real_fit = fit(cfg, real["X_train"], real["y_train"], features)
    real_pred_real = real_fit["pipe"].predict(real["X_test"])
    trtr_score = score(real["y_test"], real_pred_real)
    rows = [{
        "comparison": "TRTR", 
        "train_set": "real", 
        "test_set": "real", 
        "eps": None, 
        "seed": None,
        "n_train": len(real["X_train"]), 
        **trtr_score,
    }]

    ## Iterate through synthetic datasets
    for _, pair in pairs.iterrows():
        
        ## Load synthetic datasets for each epsilon and seed
        eps, seed = pair["eps"], pair["seed"]
        syn = load_split_synthetic.load_pair(cfg, pair, features)

        ## Predict for TRTS: train real, test synthetic
        real_pred_syn = real_fit["pipe"].predict(syn["X_test"])
        trts_score = score(syn["y_test"], real_pred_syn)
        rows.append({
            "comparison": "TRTS", 
            "train_set": "real", 
            "test_set": "synthetic", 
            "eps": eps,
            "seed": seed, 
            "n_train": len(real["X_train"]),
            **trts_score,
        })

        # Fit and predict for TSTR: train synthetic, test real
        syn_fit = fit(cfg, syn["X_train"], syn["y_train"], features)
        syn_pred_real = syn_fit["pipe"].predict(real["X_test"])
        tstr_score = score(real["y_test"], syn_pred_real)
        rows.append({
            "comparison": "TSTR",
            "train_set": "synthetic", 
            "test_set": "real",
            "eps": eps, 
            "seed": seed,
            "n_train": len(syn["X_train"]), 
            **tstr_score
        })

        # Predict for TSTS: train synthetic, test synthetic (test from same dataset as train)
        syn_pred_syn = syn_fit["pipe"].predict(syn["X_test"])
        tsts_score = score(syn["y_test"], syn_pred_syn)
        rows.append({
            "comparison": "TSTS", 
            "train_set": "synthetic", 
            "test_set": "synthetic",
            "eps": eps, 
            "seed": seed,
            "n_train": len(syn["X_train"]), 
            **tsts_score
        })

    # Convert to dataframe and order by epsilon
    raw = pd.DataFrame(rows)
    eps_order = list(pd.unique(pairs["eps"])) if len(pairs) else []
    summary = summarise(raw, eps_order)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_results_path, index=False)
    summary.to_csv(summary_path, index=False)


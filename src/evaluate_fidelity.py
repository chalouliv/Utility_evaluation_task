"""
Fidelity analysis worked into the utility analysis framework.
Calculates univariate and bivariate fidelity metrics across runs and saves the
means and standard deviations.
    Univariate: KS and TVD
    Bivariate:  Correlation similarity and Contingency similarity
Adapted from CVAE_v1/fidelity/evaluate_fidelity.py
"""
import numpy as np
import pandas as pd
from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
from sdmetrics.single_column import KSComplement, TVComplement

from src import load_real, load_split_synthetic

NAME = "evaluate_fidelity"

## For comparison between categorical and numerical columns, need to bin numerical data
## Make n bins of equal width based on values in the real data (can look at making it quantile)
def bins_from_real(real_col: pd.Series, synth_col: pd.Series, n_bins: int) -> tuple[pd.Series, pd.Series, np.ndarray]:
    edges = np.histogram_bin_edges(real_col, bins=n_bins)  ## find edges of bins based on real data
    synth_clipped = synth_col.clip(edges[0], edges[-1])  ## synthetic values outside the real range go to the lowest/highest bin
    real_binned = pd.cut(real_col, edges, include_lowest=True, labels=False).astype(str)
    synth_binned = pd.cut(synth_clipped, edges, include_lowest=True, labels=False).astype(str)
    return real_binned, synth_binned, edges


def binned_df(real_df: pd.DataFrame, synth_df: pd.DataFrame, categ: list[str], n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ## binned numerical columns for contingency score with categorical
    real_binned = pd.DataFrame(index=real_df.index)
    synth_binned = pd.DataFrame(index=synth_df.index)
    edges = {}  ## numerical column name -> array of n_bins + 1 bin edges
    for c in real_df.columns:
        if c in categ:
            real_binned[c] = real_df[c].astype(str)
            synth_binned[c] = synth_df[c].astype(str)
        else:
            real_binned[c], synth_binned[c], edges[c] = bins_from_real(real_df[c], synth_df[c], n_bins)
    return real_binned, synth_binned, edges


def score_one_dimensional(real: pd.DataFrame, synth: pd.DataFrame, categ: list[str]) -> pd.DataFrame:
    rows = []
    for c in real.columns:
        if c in categ:
            metric, score = "TVComplement", TVComplement.compute(real_data=real[c], synthetic_data=synth[c])
        else:
            metric, score = "KSComplement", KSComplement.compute(real_data=real[c], synthetic_data=synth[c])
        rows.append({"column": c, "type": "categorical" if c in categ else "numerical",
                     "metric": metric, "score": float(score)})
    return pd.DataFrame(rows)


def score_two_dimensional(real: pd.DataFrame, synth: pd.DataFrame, categ: list[str], coefficient: str, n_bins: int) -> pd.DataFrame:
    columns = list(real.columns)
    real_binned, synth_binned, _ = binned_df(real, synth, categ, n_bins)
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:  ## Upper triangle, columns not paired with themselves
            a_num = a not in categ
            b_num = b not in categ
            if a_num and b_num:
                metric = f"CorrelationSimilarity ({coefficient})"
                score = CorrelationSimilarity.compute(
                    real_data=real[[a, b]],
                    synthetic_data=synth[[a, b]],
                    coefficient=coefficient,
                )
                real_corr = real[a].corr(real[b], method=coefficient.lower())
                synth_corr = synth[a].corr(synth[b], method=coefficient.lower())
            else:  ## at least one column is categorical
                metric = "ContingencySimilarity"
                score = ContingencySimilarity.compute(
                    real_data=real_binned[[a, b]],
                    synthetic_data=synth_binned[[a, b]],
                )
                real_corr = synth_corr = np.nan

            rows.append({
                "column_1": a,
                "column_2": b,
                "metric": metric,
                "score": float(score),
                "real_correlation": real_corr,
                "synthetic_correlation": synth_corr,
            })
    return pd.DataFrame(rows)


def to_long(uni: pd.DataFrame, bi: pd.DataFrame) -> pd.DataFrame:
    ## Put univariate and bivariate scores in one table with a shared "feature" column
    uni = uni.assign(
        dimension="univariate",
        feature=uni["column"],
        column_1=uni["column"],
        column_2="",
        real_correlation=np.nan,
        synthetic_correlation=np.nan,
    ).drop(columns="column")
    bi = bi.assign(
        dimension="bivariate",
        feature=bi["column_1"] + " | " + bi["column_2"],
        type="pair",
    )
    return pd.concat([uni, bi], ignore_index=True)


def run(cfg, dpath) -> None:
    ## score between real train and each synthetic train dataset, save mean and std of each run across each eps
    out_dir = dpath / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"

    coef = cfg["fidelity"]["coefficient"]  ## NB: key spelled as in config
    n_bins = cfg["fidelity"]["n_bins"]
    baseline = cfg["fidelity"]["baseline"]  ## not used yet
    categorical = [name for name, spec in cfg["data"]["columns"].items() if spec["type"] == "categorical"]

    if raw_results_path.exists() and summary_path.exists():
        print(f"[{NAME}] {raw_results_path} and {summary_path} exist, skipping. Delete them to rerun.")
        return

    ## Load real data
    real = load_real.load(cfg)
    real = real["train"]

    ## Gather synthetic data
    pairs = load_split_synthetic.list_pairs(cfg, dpath)

    ## Iterate through synthetic datasets, scoring each (eps, seed) run
    scores = []
    for _, pair in pairs.iterrows():
        eps, seed = pair["eps"], pair["seed"]
        syn = pd.read_csv(pair["train"])
        syn = syn[real.columns]  ## same column order as real

        uni = score_one_dimensional(real, syn, categorical)
        bi = score_two_dimensional(real, syn, categorical, coef, n_bins)

        run_scores = to_long(uni, bi).assign(eps=eps, seed=seed)
        scores.append(run_scores)
    raw = pd.concat(scores, ignore_index=True)
    raw = raw[["eps", "seed", "dimension", "feature", "column_1", "column_2", "type",
                       "metric", "score", "real_correlation", "synthetic_correlation"]]
    raw.to_csv(raw_results_path, index=False)

    ## Summary: mean and std across runs and features, for each eps and metric
    summary = (
        raw.groupby(["eps", "dimension", "metric"], sort=True)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            n_features=("feature", "nunique"),
            n_runs=("seed", "nunique"),
            n_scores=("score", "size"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print(f"[{NAME}] Wrote {raw_results_path}, {summary_path}")
    print(f"Done {NAME}.")
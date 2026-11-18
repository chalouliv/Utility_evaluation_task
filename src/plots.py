import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
NAME = "plots"
 
## Color and labelling
SERIES = {
    "TSTR": {"color": "#d62ac8", "marker": "o", "label": "TSTR"},
    "TSTS": {"color": "#34cdeb", "marker": "o", "label": "TSTS"},
    "TRTS": {"color": "#ff8913", "marker": "o", "label": "TRTS"},
}
BASELINE_COLOR = "#191515" 
NULL_COLOR = "#73736e" 
 
GRID_COLOR = "#e4e3dd"
METRIC_LABELS = {
    "mae": "MAE (years)",
    "rmse": "RMSE (years)",
    "r2": "R²",
    "f1": "F1 score",
    "roc_auc": "ROC-AUC",
    "balanced_accuracy": "Balanced accuracy"
}
TARGET_LABELS = {"AGE": "Age", "PTGENDER": "Sex"}  # used in figure titles; falls back to the column name
## Which metrics to plot for each kind of target (whichever has values in raw_results)
METRIC_SETS = [["mae", "rmse", "r2"], ["f1", "roc_auc", "balanced_accuracy"]]
## Invert metrics where lower is better
INVERTED = {"mae", "rmse"}
 
## Epsilon as a label: numbers tidied (1, 1.0 and "1" -> "1"), anything else (e.g. "none") kept as text
def eps_key(e) -> str:
    try:
        return f"{float(e):g}"
    except (TypeError, ValueError):
        return str(e).strip().lower()
 
## Positioning of epsilons
def eps_positions(eps: list[str]) -> dict[str, int]:
    return {e: i for i, e in enumerate(eps)}
 
## Axis styling
def style_axes(ax, eps: list[str], pos: dict) -> None:
    ax.set_xticks([pos[e] for e in eps])
    ax.set_xticklabels(eps)
    ax.set_xlabel("Privacy budget ε")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
 
 
## Mean and std across runs (seeds)
def seed_stats(df: pd.DataFrame, col: str, eps: list[str]) -> pd.DataFrame:
    stats = df.groupby("eps")[col].agg(["mean", "std"])
    return stats.reindex([e for e in eps if e in stats.index])
 
 
## Saving plots
def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {path.name}")
 
 
def plot_metrics(raw, target, metrics, eps, pos, path) -> None:
    ## Plot performance metrics for one target side by side
    raw = raw[raw["target"] == target]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5), squeeze=False)
 
    for ax, metric in zip(axes.flat, metrics):
        for comp in ("TSTR", "TSTS", "TRTS"):  # can add back "TSTS", "TRTS"
            stats = seed_stats(raw[raw["comparison"] == comp], metric, eps)
            s = SERIES[comp]
            ax.errorbar(
                [pos[e] for e in stats.index], stats["mean"], yerr=stats["std"],
                color=s["color"], marker=s["marker"], markersize=8, linewidth=2,
                capsize=3, label=s["label"],
            )
        ## Reference lines: TRTR baseline and null (shuffled-label) model
        for comp, color in (("TRTR", BASELINE_COLOR), ("NULL", NULL_COLOR)):
            vals = raw.loc[raw["comparison"] == comp, metric].dropna()
            if len(vals):
                val = vals.mean()  # mean in case there are several null runs/seeds
                ax.axhline(val, color=color, linestyle="--", linewidth=1.5,
                           label=f"{comp} ({val:.3g})")
        style_axes(ax, eps, pos)
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric in INVERTED:
            ax.invert_yaxis()
            ax.set_title(f"{METRIC_LABELS[metric]} (axis inverted)", fontsize=10)
        else:
            ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1, 1))
 
    n_runs = raw.loc[raw["comparison"] == "TSTR"].groupby("eps").size().max()
    label = TARGET_LABELS.get(target, target)
    fig.suptitle(f"{label} Prediction Performance versus Privacy Budget (mean ± sd across {n_runs} runs)")
    save(fig, path)
 
 
def plot_coefs(coefs, target, eps, path) -> None:
    ## Forest plot: one row per feature, real coefficient vs synthetic (mean ± sd per ε)
    c = coefs[(coefs["target"] == target) & (coefs["feature"] != "(intercept)")]
    real = c[c["comparison"] == "TRTR"].set_index("feature")["coef"]
    null = c[c["comparison"] == "NULL"].set_index("feature")["coef"]
    syn = c[c["comparison"] == "TSTR"].groupby(["eps", "feature"])["coef"].agg(["mean", "std"])
 
    order = real.abs().sort_values().index  # largest real effects at the top
    y = np.arange(len(order))
    eps = [e for e in eps if e in syn.index.get_level_values("eps")]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(eps)))
    step = 0.8 / max(len(eps), 1)  # small vertical offset so ε points don't overlap
 
    fig, ax = plt.subplots(figsize=(12, 0.35 * len(order) + 2))
    for k, (e, color) in enumerate(zip(eps, colors)):
        s = syn.loc[e].reindex(order)
        ax.errorbar(s["mean"], y + (k - (len(eps) - 1) / 2) * step, xerr=s["std"],
                    fmt="o", color=color, markersize=2, capsize=1, label=f"ε = {e}")
    ax.scatter(real[order], y, marker="D", color=BASELINE_COLOR, zorder=3, label="TRTR (real)", s=10)
    ax.scatter(null.reindex(order), y, marker="x", color=NULL_COLOR, zorder=2, label="NULL", s=10)
    ax.axvline(0, color=NULL_COLOR, linewidth=0.8)
 
    ax.set_yticks(y, order)
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1, 1), loc="upper left")
 
    label = TARGET_LABELS.get(target, target)
    fig.suptitle(f"{label} Model Coefficients: Real vs Synthetic (TSTR mean ± sd across runs)")
    save(fig, path)


def plot_zeros(coefs, target, eps, path) -> None:
    ## Heatmap: share of runs where each feature's coefficient is exactly zero (L1 dropped it)
    c = coefs[(coefs["target"] == target) & (coefs["feature"] != "(intercept)")].copy()
    c["zero"] = c["coef"].abs() < 1e-10

    real = c[c["comparison"] == "TRTR"].set_index("feature")["coef"]
    order = real.abs().sort_values(ascending=False).index  # same top-to-bottom order as the forest plot

    ## Columns: real fit, then each ε (share of seeds with coef == 0)
    cols = {"TRTR": c[c["comparison"] == "TRTR"].set_index("feature")["zero"].astype(float)}
    tstr = c[c["comparison"] == "TSTR"]
    for e in eps:
        z = tstr[tstr["eps"] == e]
        if len(z):
            cols[f"ε = {e}"] = z.groupby("feature")["zero"].mean()
    grid = pd.DataFrame(cols).reindex(order)

    fig, ax = plt.subplots(figsize=(1.1 * grid.shape[1] + 3, 0.35 * len(grid) + 2))
    im = ax.imshow(grid.to_numpy(), cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(grid.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else BASELINE_COLOR)
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(len(grid)), grid.index)
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Share of runs with coefficient = 0")

    label = TARGET_LABELS.get(target, target)
    fig.suptitle(f"{label} Features Dropped by L1 (coefficient = 0)")
    save(fig, path)
 
 
def run(cfg, dpath) -> None:
    ## Make out_dir
    out_dir = dpath / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
 
    ## Find and load results of eval (evaluate.py writes to dpath / "evaluate")
    raw_path = dpath / "evaluate" / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run evaluation (Step 4) first.")
    raw = pd.read_csv(raw_path, keep_default_na=False, na_values=[""])
 
    ## Epsilons as tidy labels so config and CSV values match (TRTR has no eps and stays empty)
    raw["eps"] = raw["eps"].map(lambda e: eps_key(e) if pd.notna(e) else None)
    eps = [eps_key(e) for e in cfg["data"]["eps"]]
    pos = eps_positions(eps)
 
    ## One figure per target, with whichever metric set that target has values for
    jobs = []
    for target in raw["target"].unique():
        rows = raw[raw["target"] == target]
        for metrics in METRIC_SETS:
            if all(m in rows and rows[m].notna().any() for m in metrics):
                name = f"{target.lower()}_{'_'.join(metrics)}_vs_eps.png"
                # default args bind target/metrics now; a plain lambda would use the last loop values
                jobs.append((name, lambda p, t=target, m=metrics: plot_metrics(raw, t, m, eps, pos, p)))
 
    ## One coefficient figure per target (only if evaluate.py wrote coefficients.csv)
    coefs_path = dpath / "evaluate" / "coefficients.csv"
    if coefs_path.exists():
        coefs = pd.read_csv(coefs_path, keep_default_na=False, na_values=[""])
        coefs["eps"] = coefs["eps"].map(lambda e: eps_key(e) if pd.notna(e) else None)
        for target in coefs["target"].unique():
            name = f"{target.lower()}_coefficients.png"
            jobs.append((name, lambda p, t=target: plot_coefs(coefs, t, eps, p)))

        ## Zero-coefficient heatmap for classification targets only (those with ROC-AUC in raw_results)
        clf = raw.loc[raw["roc_auc"].notna(), "target"].unique() if "roc_auc" in raw else []
        for target in clf:
            name = f"{target.lower()}_zero_coefficients.png"
            jobs.append((name, lambda p, t=target: plot_zeros(coefs, t, eps, p)))
    else:
        print(f"[plots] {coefs_path} not found, skipping coefficient plots.")
 
    for name, draw in jobs:
        path = out_dir / name
        if path.exists():
            print(f"[plots] {name} exists, skipping. Delete it to redraw.")
            continue
        draw(path)

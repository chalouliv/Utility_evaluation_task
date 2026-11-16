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
BASELINE_COLOR = "#50504b"
GRID_COLOR = "#e4e3dd"
METRIC_LABELS = {
    "mae": "MAE (years)",
    "rmse": "RMSE (years)",
    "r2": "R²",
    "f1": "F1 score",
    "roc_auc": "ROC-AUC",
    "log_loss": "Log loss",
}
TARGET_LABELS = {"AGE": "Age", "PTGENDER": "Sex"}  # used in figure titles; falls back to the column name
## Which metrics to plot for each kind of target (whichever has values in raw_results)
METRIC_SETS = [["mae", "rmse", "r2"], ["f1", "roc_auc", "log_loss"]]
## Invert metrics where lower is better
INVERTED = {"mae", "rmse", "log_loss"}

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
        for comp in ("TSTR",):  # can add back "TSTS", "TRTS"
            stats = seed_stats(raw[raw["comparison"] == comp], metric, eps)
            s = SERIES[comp]
            ax.errorbar(
                [pos[e] for e in stats.index], stats["mean"], yerr=stats["std"],
                color=s["color"], marker=s["marker"], markersize=8, linewidth=2,
                capsize=3, label=s["label"],
            )
        trtr = raw.loc[raw["comparison"] == "TRTR", metric]
        if len(trtr) and pd.notna(trtr.iloc[0]):
            ax.axhline(trtr.iloc[0], color=BASELINE_COLOR, linestyle="--", linewidth=1.5,
                       label=f"TRTR ({trtr.iloc[0]:.3g})")
        style_axes(ax, eps, pos)
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric in INVERTED:
            ax.invert_yaxis()
            ax.set_title(f"{METRIC_LABELS[metric]} (axis inverted)", fontsize=10)
        else:
            ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.legend(frameon=False, fontsize=8)

    n_runs = raw.loc[raw["comparison"] == "TSTR"].groupby("eps").size().max()
    label = TARGET_LABELS.get(target, target)
    fig.suptitle(f"{label} Prediction Performance versus Privacy Budget (mean ± sd across {n_runs} runs)")
    save(fig, path)


def run(cfg, dpath) -> None:
    ## Make out_dir
    out_dir = dpath / NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    ## Find and load results of eval (evaluate.py writes to dpath / "evaluate")
    raw_path = dpath / "evaluate" / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run evaluation (Step 4) first.")
    raw = pd.read_csv(raw_path)

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

    for name, draw in jobs:
        path = out_dir / name
        if path.exists():
            print(f"[plots] {name} exists, skipping. Delete it to redraw.")
            continue
        draw(path)

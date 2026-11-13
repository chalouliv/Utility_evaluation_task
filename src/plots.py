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
    "TRTS": {"color": "#ff8913", "marker": "o", "label": "TRTS"}
}
BASELINE_COLOR = "#50504b"
GRID_COLOR = "#e4e3dd"
METRIC_LABELS = {
    "mae": "MAE (years)",
    "rmse": "RMSE (years)",
    "r2": "R²"
}
 
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
 
## Invert metrics where lower is better
INVERTED = {"mae", "rmse"}

def plot_metrics(raw, metrics, eps, pos, path) -> None:
    
    ## Plot preformance metrics side by side (3 in this case)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5), squeeze=False)

    ## Very fancy plotting function written by Claude 
    for ax, metric in zip(axes.flat, metrics):
        for comp in ("TSTR",):  # can add back "TSTS", "TRTS"
            stats = seed_stats(raw[raw["comparison"] == comp], metric, eps)
            s = SERIES[comp]
            ax.errorbar(
                [pos[e] for e in stats.index], stats["mean"], yerr=stats["std"],
                color=s["color"], marker=s["marker"], markersize=8, linewidth=2,
                capsize=3, label=s["label"],
            )
        trtr = raw.loc[raw["comparison"] == "TRTR", metric].iloc[0]
        ax.axhline(trtr, color=BASELINE_COLOR, linestyle="--", linewidth=1.5,
                   label=f"TRTR ({trtr:.3g})")
        style_axes(ax, eps, pos)
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric in INVERTED:
            ax.invert_yaxis()
            ax.set_title(f"{METRIC_LABELS[metric]} (axis inverted)", fontsize=10)
        else:
            ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Age Prediction Performance versus Privacy Budget (mean ± sd across 5 runs)")
    save(fig, path)
 
def run(cfg, dpath) -> None:
    ## Make out_dir
    out_dir = dpath / NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    ## Find and load results of eval
    eval_dir = Path(cfg["run"]["output_dir"]) / "evaluate"
    raw_path = eval_dir / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run evaluation (Step 4) first.")
    raw = pd.read_csv(raw_path, dtype={"eps": "string"})

    ## Get epsilon postitioning 
    eps = [str(e) for e in cfg["data"]["eps"]]
    pos = eps_positions(eps)
 
    jobs = [] ## Will add more later 
    jobs.append(("mae_rmse_r2_vs_eps.png",
                 lambda p: plot_metrics(raw, ["mae", "rmse", "r2"], eps, pos, p)))
 
    for name, draw in jobs:
        path = out_dir / name
        if path.exists():
            print(f"[plots] {name} exists, skipping. Delete it to redraw.")
            continue
        draw(path)

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

NAME = "load_split_synthetic"

def list_pairs(cfg, dpath: Path) -> pd.DataFrame:
    ## Return list of synthetic train/test pairs 
    records = []
    out_dir = dpath / "synth_data"
    pattern = cfg["data"]["synth_file_pattern"]

    for eps in cfg["data"]["eps"]:
        for seed in cfg["data"]["seeds"]:
            run_name = pattern.format(eps=eps, seed=seed)
            train = out_dir / f"{run_name}_train.csv"
            test = out_dir / f"{run_name}_test.csv"
            records.append({"eps": eps, "seed": seed, "train": train, "test": test})

   # print(records[1:5,:])

    return pd.DataFrame(records)

### CHECK IF NEEDED, CAN PROBABLY BE REPLACED BY GET_XY IN LOAD_REAL
def load_pair(cfg, target, row, features: list[str]) -> dict:
    ## Load one synthetic train/test pair, seperate out a given target (for evaluate.py)
    out = {}
    for split in ("train", "test"):
        df = pd.read_csv(row[split])
        features = [c for c in df.columns if c != target]
        X, y = df[features], df[target]
        out[f"X_{split}"], out[f"y_{split}"] = X, y
    return out


def run(cfg, dpath: Path) -> pd.DataFrame:
    ## Create paths for outputs and check if they already exist 
    out_dir = dpath / "synth_data"
    if out_dir.exists(): ## do not reload and split
        print(f"[{NAME}] {out_dir} exists, skipping. Delete it to rerun.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    synth_dir = Path(cfg["paths"]["synth_dir"])
    run_seed = cfg["run"]["seed"]
    pattern = cfg["data"]["synth_file_pattern"]

    ## Load each synthetic sets (following pattern in config), split into training and testing
    for eps in cfg["data"]["eps"]: 
        for seed in cfg["data"]["seeds"]:
            ## Loading synthetic set
            run_name = pattern.format(eps=eps, seed=seed) 
            synth = pd.read_csv(synth_dir / run_name / "generation" / "synthetic.csv")

            ## split into train and test
            train, test = train_test_split(synth, test_size=0.2, random_state=run_seed)

            ## save splits
            train.to_csv(out_dir / f"{run_name}_train.csv", index=False)
            test.to_csv(out_dir / f"{run_name}_test.csv", index=False)
    
    pairs = list_pairs(cfg, dpath)
    # counts = pairs.groupby("eps", sort=False)["seed"].nunique()
    # print(f"[load_synthetic] {len(pairs)} train/test pairs. Seeds per ε: {counts.to_dict()}")
    return pairs
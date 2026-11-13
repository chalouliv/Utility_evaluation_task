from pathlib import Path
import pandas as pd

NAME = "load_real"

def load(config) -> dict:
    ## Load real data. Returns the dataframes and feature order.
    target = config["data"]["target"]
    train = pd.read_csv(config["paths"]["real_train"])
    test = pd.read_csv(config["paths"]["real_train"])
    return {
        "train": train,
        "test": test    
    }

def run(config: dict, dpath: Path) -> None:
    real = load(config)
    print(
         f"[{NAME}] train {real['X_train'].shape}, test {real['X_test'].shape}, "
         f"{len(real['features'])} features"
    )
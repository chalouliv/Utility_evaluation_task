from pathlib import Path
import pandas as pd

NAME = "load_real"

def load(config) -> dict:
    ## Load real data. Returns the dataframes and feature order.
    target = config["data"]["target"]
    train = pd.read_csv(config["paths"]["real_train"])
    test = pd.read_csv(config["paths"]["real_train"])
    features = [c for c in train.columns if c != target]

    ## Seperate target and order features
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    return {
        "features": features,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

def run(config: dict, dpath: Path) -> None:
    real = load(config)
    # print(
    #     f"[{NAME}] train {real['X_train'].shape}, test {real['X_test'].shape}, "
    #     f"{len(real['features'])} features"
    # )
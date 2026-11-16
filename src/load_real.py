from pathlib import Path
import pandas as pd

NAME = "load_real"

def load(cfg) -> dict:
    ## Load real data. Returns the dataframes and feature order.
    train = pd.read_csv(cfg["paths"]["real_train"])
    test = pd.read_csv(cfg["paths"]["real_test"])
    return {
        "train": train,
        "test": test    
    }

def get_xy(data: dict, split:str, target:str) -> dict: 
    df = data[split]
    features = [c for c in df.columns if c != target]
    return {
        "X" : df[features], 
        "y" : df[target],
        "features" : features, 
        "target" : target
    }

def run(cfg, dpath: Path): 
    real = load(cfg)
    return print(f"Loaded real data trainset of size {real["train"].info()} and testset of size {real["test"].info()})")
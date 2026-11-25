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

def get_xy(data: dict, split:str, target:str, task:str) -> dict: 
    ## todo: Make encoding generalizable, make sure features are all in the same order 
    ## Split data into x and y and encodes if y is classification
    df = data[split]
    features = [c for c in df.columns if c != target]
    y = df[target]
    if task == "classification":
       y = y.map({"Male": 1, "Female": 0}).rename(target)
    return {
        "X" : df[features], 
        "y" : y,
        "features" : features, 
        "target" : target
    }

def run(cfg, dpath): 
    real = load(cfg)
    print(f"Loaded real data trainset of size {real["train"].info()} and testset of size {real["test"].info()})")
    return print(f"Done {NAME}.")
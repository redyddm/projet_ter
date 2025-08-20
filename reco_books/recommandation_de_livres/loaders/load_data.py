import pandas as pd
from pathlib import Path

def load_csv(path: Path):
    return pd.read_csv(path)

def load_pkl(path: Path):
    return pd.read_pickle(path)
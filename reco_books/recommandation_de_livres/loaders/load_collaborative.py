import pandas as pd
from pathlib import Path

def load_books(path: Path):
    return pd.read_csv(path)

def load_ratings(path: Path):
    return pd.read_csv(path)

def load_users(path: Path):
    return pd.read_csv(path)
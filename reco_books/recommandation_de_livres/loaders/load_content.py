import pandas as pd
from pathlib import Path

def load_books(path: Path):
    return pd.read_csv(path)

def load_authors(path: Path):
    return pd.read_csv(path)

def load_categories(path: Path):
    return pd.read_csv(path)
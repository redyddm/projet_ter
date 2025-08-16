import os
import pandas as pd


def save_df_to_csv(df: pd.DataFrame, filepath: str, index: bool = False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index, encoding="utf-8")


def save_df_to_pickle(df: pd.DataFrame, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_pickle(filepath)

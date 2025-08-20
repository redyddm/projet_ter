import os
import pandas as pd
import requests


def save_df_to_csv(df: pd.DataFrame, filepath: str, index: bool = False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index, encoding="utf-8")


def save_df_to_pickle(df: pd.DataFrame, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_pickle(filepath)

def get_cover_url(isbn):
    url = f"https://bookcover.longitood.com/bookcover/{isbn}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("url")
        else:
            return None
    except:
        return None
    
def stars(rating: float, max_stars: int = 5) -> str:
    """Retourne une chaîne d'étoiles ⭐ pour une note donnée."""
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    empty_stars = max_stars - full_stars - int(half_star)

    return "⭐" * full_stars + ("✨" if half_star else "") + "☆" * empty_stars

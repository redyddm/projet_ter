from pathlib import Path
import pandas as pd
import pickle
import typer
from loguru import logger
from tqdm import tqdm

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.iads.collabo_utils import recommandation_collaborative_top_k
from recommandation_de_livres.loaders.load_data import load_parquet

app = typer.Typer()

choice = input("Choix du dataset [2] : Recommender (1), Goodreads (2) ") or "2"

if choice == "1":
    DIR = "recommender"
elif choice == "2":
    DIR = "goodreads"
else:
    raise ValueError("Choix invalide (1 ou 2 attendu)")

@app.command()
def main(
    user_index: str,
    model_path: Path = MODELS_DIR / DIR / "nmf_model.pkl",
    ratings_path: Path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet",
    top_k: int = 5
):
    
    logger.info("Loading ratings and books data...")
    ratings = load_parquet(ratings_path)
    user_ratings = ratings.loc[ratings['user_index']==int(user_index)].iloc[0]
    if user_ratings.empty:
        raise ValueError(f"Aucun utilisateur trouvé avec user_index={user_index}")
    else:
        user_id = user_ratings['user_id']
    ratings["user_id"] = ratings["user_id"].astype(str)

    logger.info(f"Loading trained NMF model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if user_id not in ratings['user_id'].unique():
        logger.warning(f"User {user_id} not found in ratings dataset.")
        return

    logger.info(f"Predicting top-{top_k} recommendations for user {user_id}...")

    top_recommendations = recommandation_collaborative_top_k(
    k=top_k,
    user_id=user_id,
    model=model,
    ratings=ratings
    )

    logger.info("Top recommendations:")
    logger.info("\n" + top_recommendations[['title', 'authors']].to_string(index=False))

if __name__ == "__main__":
    app()
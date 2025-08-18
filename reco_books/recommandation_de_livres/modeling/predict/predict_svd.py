from pathlib import Path
import pandas as pd
import pickle
import typer
from loguru import logger
from tqdm import tqdm

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from reco_books.recommandation_de_livres.iads.svd_utils import recommandation_collaborative_top_k

app = typer.Typer()

@app.command()
def main(
    user_id: int,
    model_path: Path = MODELS_DIR / "svd_model.pkl",
    ratings_path: Path = PROCESSED_DATA_DIR / "collaborative_dataset.csv",
    top_k: int = 5
):
    
    logger.info("Loading ratings and books data...")
    ratings = pd.read_csv(ratings_path)

    logger.info(f"Loading trained SVD model from {model_path}")
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
    logger.info("\n" + top_recommendations.to_string(index=False))


if __name__ == "__main__":
    app()

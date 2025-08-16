from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from loaders import load_collaborative
from dataset import build_collaborative_dataset
from reco_books.recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

DIR = 'Recommender_dataset'

@app.command()
def main(
    books_path: Path = RAW_DATA_DIR / DIR / "Books.csv",
    ratings_path: Path = RAW_DATA_DIR / DIR / "Ratings.csv",
    output_path: Path = PROCESSED_DATA_DIR / "collaborative_dataset.csv",
    output_path_pkl: Path = PROCESSED_DATA_DIR / "collaborative_dataset.pkl",
):
    logger.info("Loading raw datasets...")
    books = load_collaborative.load_books(books_path)
    ratings = load_collaborative.load_ratings(ratings_path)

    logger.info("Building a collaborative filtering dataset...")
    ratings_df = build_collaborative_dataset.build_collaborative_dataset(books, ratings)

    logger.info(f"Saving processed dataset to {output_path} and {output_path_pkl}")
    save_df_to_csv(ratings_df, output_path)
    save_df_to_pickle(ratings_df, output_path_pkl)
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from reco_books.recommandation_de_livres.loaders import load_data
from recommandation_de_livres.build_dataset import build_user_books_dataset
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle
from recommandation_de_livres.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

DIR = 'Recommender_dataset'
DIR_OUTPUT = 'recommender'

@app.command()
def main(
    books_path: Path = RAW_DATA_DIR / DIR / "Books.csv",
    ratings_path: Path = RAW_DATA_DIR / DIR / "Ratings.csv",
    users_path: Path = RAW_DATA_DIR / DIR / "Users.csv",
    output_path: Path = PROCESSED_DATA_DIR / DIR_OUTPUT / "user_book_dataset.csv",
    output_path_pkl: Path = PROCESSED_DATA_DIR / DIR_OUTPUT / "user_book_dataset.pkl",
):
    logger.info("Loading raw datasets...")
    books = load_data.load_books(books_path)
    ratings = load_data.load_ratings(ratings_path)
    users = load_data.load_users(users_path)

    logger.info("Building users-books dataset...")
    books_df = build_user_books_dataset.build_user_book_dataset(books, ratings, users)

    logger.info(f"Saving processed dataset to {output_path} and {output_path_pkl}")

    save_df_to_csv(books_df, output_path)
    save_df_to_pickle(books_df, output_path_pkl)
    
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()

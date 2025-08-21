from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from recommandation_de_livres.loaders import load_data
from recommandation_de_livres.build_dataset import build_collabo_dataset_gdr
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle, save_df_to_parquet
from recommandation_de_livres.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

DIR = 'goodreads'

@app.command()
def main(
    books_path: Path = INTERIM_DATA_DIR / DIR / "books_authors.csv",
    ratings_path: Path = RAW_DATA_DIR / DIR / "reviews.csv",
    output_path: Path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.csv",
    output_path_pkl: Path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.pkl",
    output_path_parquet: Path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet",
):
    logger.info("Loading raw datasets...")
    books = load_data.load_csv(books_path)
    ratings = load_data.load_csv(ratings_path)

    logger.info("Building a collaborative filtering dataset...")
    ratings_df = build_collabo_dataset_gdr.build_collaborative_dataset(books, ratings)

    logger.info(f"Saving processed dataset to {output_path} and {output_path_pkl} and {output_path_parquet}")
    save_df_to_csv(ratings_df, output_path)
    save_df_to_pickle(ratings_df, output_path_pkl)
    save_df_to_parquet(ratings_df, output_path_parquet)
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()

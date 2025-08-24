from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from recommandation_de_livres.loaders import load_data
from recommandation_de_livres.build_dataset import build_collaborative_dataset
from recommandation_de_livres.iads.create_users import create_users_file
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

dataset_choice = input("Choix du dataset [2] : Recommender (1), Goodreads (2) ") or "2"

if dataset_choice == "1":
    DIR = "recommender"
    authors=None
elif dataset_choice == "2":
    DIR = "goodreads"
    authors_path: Path = RAW_DATA_DIR / DIR / "authors.csv"
    authors=load_data.load_csv(authors_path)
else:
    raise ValueError("Choix invalide (1 ou 2 attendu)")

@app.command()
def main(
    books_path: Path = INTERIM_DATA_DIR / DIR / "books_uniform.parquet",
    ratings_path: Path = INTERIM_DATA_DIR / DIR / "ratings_uniform.parquet",
    output_path: Path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.csv",
    output_path_parquet: Path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet",
    output_users: Path =PROCESSED_DATA_DIR / DIR / "users.csv"
):
    
    logger.info("Loading raw datasets...")
    books = load_data.load_parquet(books_path)
    ratings = load_data.load_parquet(ratings_path)

    logger.info("Building a collaborative filtering dataset...")
    ratings_df = build_collaborative_dataset.build_collaborative_dataset(books, ratings, authors=authors, min_ratings=0, min_users_interaction=100)

    logger.info(f"Saving processed dataset to {output_path} and {output_path_parquet}")
    create_users_file(output_path, output_users)
    save_df_to_csv(ratings_df, output_path)
    save_df_to_parquet(ratings_df, output_path_parquet)
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()

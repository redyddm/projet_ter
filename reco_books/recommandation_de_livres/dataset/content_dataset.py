from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import numpy as np

from recommandation_de_livres.loaders import load_data
from recommandation_de_livres.build_dataset import build_content_dataset
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle, save_df_to_parquet
from recommandation_de_livres.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

dataset_choice = input("Choix du dataset [2] : Recommender (1), Goodreads (2) ") or "2"

if dataset_choice == "1":
    DIR = "recommender"
elif dataset_choice == "2":
    DIR = "goodreads"
else:
    raise ValueError("Choix invalide (1 ou 2 attendu)")

@app.command()
def main(
    books_path: Path = INTERIM_DATA_DIR / DIR / "books_uniform.parquet",
    authors_path: Path = None,
    categories_path: Path = None,
    output_path_csv: Path = PROCESSED_DATA_DIR / DIR / "content_dataset.csv",
    output_path_parquet: Path = PROCESSED_DATA_DIR / DIR / "content_dataset.parquet",
):
    logger.info(f"Loading raw datasets for '{DIR}'...")
    books = load_data.load_parquet(books_path)

    if dataset_choice == "1":
        description = None
    elif dataset_choice == "2":
        authors_path: Path = RAW_DATA_DIR / DIR / "authors.csv"
        categories_path: Path = RAW_DATA_DIR / DIR / "categories.csv"
        description = "description"

    add_lang_input = input("Ajouter les langues [2] : True (1), False (2) ") or "2"
    add_lang = True if add_lang_input=="1" else False

    allowed_langs=['en', 'eng', 'en-US', 'en-GB', 'en-CA']

    authors = load_data.load_csv(authors_path) if authors_path else None
    categories = load_data.load_csv(categories_path) if categories_path else None

    logger.info("Building content-based dataset...")
    books_df = build_content_dataset.build_content_dataset(
        books=books,
        authors=authors,
        categories=categories,
        lang_col="language",
        desc_col=description,
        dataset_dir=INTERIM_DATA_DIR / DIR, 
        allowed_langs=allowed_langs,
        add_language=add_lang
    )

    logger.info(f"Saving processed dataset to {output_path_csv} and {output_path_parquet}")
    save_df_to_csv(books_df, output_path_csv)
    save_df_to_parquet(books_df, output_path_parquet)
    
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()

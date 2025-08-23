from pathlib import Path
from loguru import logger
import typer
import pandas as pd

from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from recommandation_de_livres.preprocessing.preprocess_uniform import rename_ratings_columns, rename_books_columns

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
    books_path: Path = None,
    ratings_path: Path = None,
    output_ratings_csv: Path = None,
    output_ratings_parquet: Path = None,
    output_books_csv: Path = None,
    output_books_parquet: Path = None
):
    
    books_path = books_path or RAW_DATA_DIR / DIR / "books.csv"
    ratings_path = ratings_path or RAW_DATA_DIR / DIR / "ratings.csv"
    output_ratings_csv = output_ratings_csv or INTERIM_DATA_DIR / DIR / "ratings_uniform.csv"
    output_ratings_parquet = output_ratings_parquet or INTERIM_DATA_DIR / DIR / "ratings_uniform.parquet"
    output_books_csv = output_books_csv or INTERIM_DATA_DIR / DIR / "books_uniform.csv"
    output_books_parquet = output_books_parquet or INTERIM_DATA_DIR / DIR / "books_uniform.parquet"

    logger.info(f"Loading datasets for '{DIR}'...")
    ratings = pd.read_csv(ratings_path)
    books = pd.read_csv(books_path)

    logger.info("Uniformisation des datasets...")
    
    if dataset_choice == "1":
        ratings_uniform = rename_ratings_columns(
            ratings,
            user_col="User-ID",
            item_col="ISBN",
            rating_col="Book-Rating"
        )

        logger.info("Uniformisation du dataset books...")
        books_uniform = rename_books_columns(
            books,
            isbn_col="ISBN",
            book_id_col=None,
            title_col="Book-Title",
            author_col="Book-Author",
            publisher_col="Publisher",
            year_col="Year-Of-Publication",
            image_col="Image-URL-L"
        )
    
    elif dataset_choice=="2":
        ratings_uniform = rename_ratings_columns(
            ratings,
            user_col="user_id",
            item_col="book_id",
            rating_col="rating"
        )

        logger.info("Uniformisation du dataset books...")
        books_uniform = rename_books_columns(
            books,
            isbn_col="isbn",
            book_id_col="book_id",
            language_col="language_code",
            title_col="title",
            author_col="authors",
            publisher_col="publisher",
            year_col="publication_year",
            image_col="image_url"
        )
    


    # Sauvegarde
    logger.info("Sauvegarde des fichiers uniformisés...")
    save_df_to_csv(ratings_uniform, output_ratings_csv)
    save_df_to_parquet(ratings_uniform, output_ratings_parquet)
    save_df_to_csv(books_uniform, output_books_csv)
    save_df_to_parquet(books_uniform, output_books_parquet)

    logger.success("Uniformisation et sauvegarde terminées !")

if __name__ == "__main__":
    app()

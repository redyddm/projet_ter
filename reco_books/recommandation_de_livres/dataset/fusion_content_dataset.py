from pathlib import Path
from loguru import logger
import typer
import pandas as pd

from recommandation_de_livres.loaders import load_data
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    books_path_recommender: Path = PROCESSED_DATA_DIR / "recommender" / "content_dataset.parquet",
    books_path_goodreads: Path = PROCESSED_DATA_DIR / "goodreads" / "content_dataset.parquet",
    output_path_csv: Path = PROCESSED_DATA_DIR / "fusion" / "content_dataset.csv",
    output_path_parquet: Path = PROCESSED_DATA_DIR / "fusion" / "content_dataset.parquet",
):
    logger.info("Chargement des datasets processed...")

    books_recommender = load_data.load_parquet(books_path_recommender)
    books_goodreads = load_data.load_parquet(books_path_goodreads)

    # Harmoniser les colonnes (intersection)
    common_cols = list(set(books_recommender.columns) & set(books_goodreads.columns))
    books_recommender = books_recommender[common_cols]
    books_goodreads = books_goodreads[common_cols]

    # Fusion & suppression doublons ISBN si présent
    books = pd.concat([books_recommender, books_goodreads], ignore_index=True)
    if "isbn" in books.columns:
        books.drop_duplicates(subset=["isbn"], inplace=True)

    books['item_id']=books['item_id'].astype(str)

    logger.info(f"Fusion terminée : {len(books)} livres au total.")

    # Sauvegarde
    output_path_csv.parent.mkdir(parents=True, exist_ok=True)
    output_path_parquet.parent.mkdir(parents=True, exist_ok=True)
    save_df_to_csv(books, output_path_csv)
    save_df_to_parquet(books, output_path_parquet)

    logger.success(f"Dataset fusionné sauvegardé dans : {output_path_parquet}")

if __name__ == "__main__":
    app()

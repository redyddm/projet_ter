from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from reco_books.recommandation_de_livres.loaders import load_content_depository
from reco_books.recommandation_de_livres.dataset import build_content_dataset_depository
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle
from recommandation_de_livres.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

DIR = 'Book_depository'

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / DIR / "dataset.csv",
    authors_path: Path = RAW_DATA_DIR / DIR / "authors.csv",
    categories_path: Path = RAW_DATA_DIR / DIR /  "categories.csv",
    output_path: Path = PROCESSED_DATA_DIR / "content_dataset_depository.csv",
    output_path_pkl: Path = PROCESSED_DATA_DIR / "content_dataset_depository.pkl",
):
    logger.info("Loading raw datasets...")
    books = load_content_depository.load_books(input_path)
    authors = load_content_depository.load_authors(authors_path)
    categories = load_content_depository.load_categories(categories_path)

    logger.info("Building content-based dataset...")
    books_df = build_content_dataset_depository.build_content_dataset(books, authors, categories)

    logger.info(f"Saving processed dataset to {output_path} and {output_path_pkl}")
    save_df_to_csv(books_df, output_path)
    save_df_to_pickle(books_df, output_path_pkl)
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()

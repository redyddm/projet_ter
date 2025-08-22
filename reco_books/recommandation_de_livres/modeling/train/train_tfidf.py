from pathlib import Path
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
import typer

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
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
    # chemins par défaut
    features_path: Path = PROCESSED_DATA_DIR / DIR / "features_tfidf.parquet",
    model_path: Path = MODELS_DIR / DIR / "tfidf_model.pkl",
    matrix_path: Path = PROCESSED_DATA_DIR / DIR / "tfidf_matrix.pkl",
):
    logger.info("Loading the TF-IDF features dataset...")
    features_df = load_parquet(features_path)

    logger.info("Creating TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(ngram_range=(1, 3), lowercase=True, stop_words='english')

    logger.info("Fitting TF-IDF on the cleaned texts (title + authors)...")
    tfidf_matrix = tfidf.fit_transform(features_df['text_clean'])

    logger.info(f"Saving the TF-IDF model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(tfidf, f)
    logger.success("TF-IDF model saved.")

    logger.info(f"Saving the TF-IDF matrix to {matrix_path}...")
    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)
    logger.success("TF-IDF matrix saved.")

if __name__ == "__main__":
    app()

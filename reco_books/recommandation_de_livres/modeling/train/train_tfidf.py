from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
import typer
import pickle

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    # ---- chemins par défaut ----
    features_path: Path = PROCESSED_DATA_DIR / "content_dataset.pkl",
    model_path: Path = MODELS_DIR / "tfidf_model.pkl",
    matrix_path: Path = PROCESSED_DATA_DIR / "tfidf_matrix.pkl",
):
    logger.info("Loading the dataset...")
    content_df = pd.read_pickle(features_path)

    logger.info("Creating TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(ngram_range=(1,5), lowercase=True, stop_words='english')

    logger.info("Fitting TF-IDF on the titles...")
    tfidf_matrix = tfidf.fit_transform(content_df['title_clean'])

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

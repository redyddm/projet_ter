from pathlib import Path
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
import typer

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

choice = input("Choix du dataset [3] : Recommender (1), Depository (2), Goodreads (3) ") or "3"

if choice == "1":
    DIR = "recommender"
elif choice == "2":
    DIR = "depository"
elif choice == "3":
    DIR = "goodreads"
else:
    raise ValueError("Choix invalide (1, 2 ou 3 attendu)")

@app.command()
def main(
    # chemins par défaut
    features_path: Path = PROCESSED_DATA_DIR / DIR / "features_tfidf.pkl",
    model_path: Path = MODELS_DIR / DIR / "tfidf_model.pkl",
    matrix_path: Path = PROCESSED_DATA_DIR / DIR / "tfidf_matrix.pkl",
):
    logger.info("Loading the TF-IDF features dataset...")
    features_df = pd.read_pickle(features_path)

    logger.info("Creating TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(ngram_range=(1, 5), lowercase=True, stop_words='english')

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

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from recommandation_de_livres.loaders.load_data import load_parquet

from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.content_utils import get_book_embedding, recommandation_content_top_k

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
    title: str = typer.Option("Harry Potter", prompt="Entrez le titre du livre", help="Titre du livre pour la recommandation"),
    top_k: int = typer.Option(5, prompt="Entrez le nombre de recommandations souhaité", help="Nombre de recommandations à afficher"),
    model_sbert_path: Path = MODELS_DIR / DIR / "sbert_model",
    embeddings_sbert_path: Path = PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy",
    content_path: Path = PROCESSED_DATA_DIR / DIR / "content_dataset.parquet",
):
    logger.info("Loading content data...")
    content_df = load_parquet(content_path)

    logger.info("Loading Sentence-BERT model...")
    model = SentenceTransformer(str(model_sbert_path))

    logger.info("Chargement des embeddings pré-calculés...")
    embeddings = np.load(embeddings_sbert_path)

    logger.info("Recherche des livres similaires...")
    top_books =recommandation_content_top_k(title, embeddings, model, content_df, k=top_k)

    logger.info("Top recommandations :")
    logger.info(f"\n{top_books[['title', 'authors']]}")
    

if __name__ == "__main__":
    app()
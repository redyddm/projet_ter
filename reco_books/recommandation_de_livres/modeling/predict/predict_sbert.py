import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.content_utils import get_book_embedding

app = typer.Typer()

@app.command()

def main(
    title: str = typer.Option("Harry Potter", prompt="Entrez le titre du livre", help="Titre du livre pour la recommandation"),
    top_k: int = typer.Option(5, prompt="Entrez le nombre de recommandations souhaité", help="Nombre de recommandations à afficher"),
    model_sbert_path: Path = MODELS_DIR / "sbert_model",
    embeddings_sbert_path: Path = PROCESSED_DATA_DIR / "embeddings_sbert.npy",
    content_path: Path = PROCESSED_DATA_DIR / "content_dataset.pkl",
):
    logger.info("Loading content data...")
    content_df = pd.read_pickle(content_path)

    logger.info("Loading Sentence-BERT model...")
    model = SentenceTransformer(str(model_sbert_path))

    logger.info("Chargement des embeddings pré-calculés...")
    embeddings = np.load(embeddings_sbert_path)

    
    query_vec = get_book_embedding(title, model)
    similarity = cosine_similarity(query_vec, embeddings)[0]

    top_indices = np.argsort(similarity)[-top_k:][::-1]
    top_books = content_df.iloc[top_indices][['title', 'authors']].copy()

    logger.info("Top recommandations :")
    logger.info(f"\n{top_books}")
    

if __name__ == "__main__":
    app()

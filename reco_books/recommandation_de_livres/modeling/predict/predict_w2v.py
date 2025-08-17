import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.metrics.pairwise import cosine_similarity
import gensim

from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.content_utils import get_book_embedding

app = typer.Typer()

@app.command()

def main(
    title: str = typer.Option("Harry Potter", prompt="Entrez le titre du livre", help="Titre du livre pour la recommandation"),
    top_k: int = typer.Option(5, prompt="Entrez le nombre de recommandations souhaité", help="Nombre de recommandations à afficher"),
    model_w2v_path: Path = MODELS_DIR / "word2vec.model",
    embeddings_w2v_path: Path = PROCESSED_DATA_DIR / "embeddings_w2v.npy",
    content_path: Path = PROCESSED_DATA_DIR / "content_dataset.pkl",
):
    logger.info("Loading content data...")
    content_df = pd.read_pickle(content_path)

    logger.info("Loading Word2Vec model...")
    model = gensim.Word2Vec.load(str(model_w2v_path))

    logger.info("Chargement des embeddings pré-calculés...")
    embeddings = np.load(embeddings_w2v_path)

    logger.info("Recherche des livres similaires...")
    query_vec = get_book_embedding(title, model)
    similarity = cosine_similarity(query_vec, embeddings)[0]

    top_indices = np.argsort(similarity)[-top_k:][::-1]
    top_books = content_df.iloc[top_indices][['title', 'authors']].copy()

    logger.info(f"Top {top_k} recommandations :")
    logger.info(f"\n{top_books}")
    




if __name__ == "__main__":
    app()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import gensim
import joblib

from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.content_utils import recommandation_content_top_k
from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.iads.utils import choose_dataset_interactively

app = typer.Typer()

DIR = choose_dataset_interactively()
print(f"Dataset choisi : {DIR}")

# --- Choix simple pour KNN ---
use_knn = input("Voulez-vous utiliser KNN pour accélérer la recherche ? (o/n) : ").strip().lower() == "o"
if use_knn:
    knn_path = MODELS_DIR / DIR / "knn_model_w2v.joblib"
    if knn_path.exists():
        knn = joblib.load(knn_path)
        print(f"KNN chargé depuis : {knn_path}")
    else:
        print("Fichier KNN non trouvé, KNN ne sera pas utilisé.")
        knn = None
else:
    knn = None

@app.command()
def main(
    title: str = typer.Option("Harry Potter", prompt="Entrez le titre du livre", help="Titre du livre pour la recommandation"),
    top_k: int = typer.Option(5, prompt="Entrez le nombre de recommandations souhaité", help="Nombre de recommandations à afficher"),
    model_w2v_path: Path = MODELS_DIR / DIR / "word2vec.model",
    embeddings_w2v_path: Path = PROCESSED_DATA_DIR / DIR / "embeddings_w2v.npy",
    content_path: Path = PROCESSED_DATA_DIR / DIR / "content_dataset.parquet",
):
    logger.info("Loading content data...")
    content_df = load_parquet(content_path)

    logger.info("Loading Word2Vec model...")
    model = gensim.models.Word2Vec.load(str(model_w2v_path))

    logger.info("Chargement des embeddings pré-calculés...")
    embeddings = np.load(embeddings_w2v_path)

    logger.info("Recherche des livres similaires...")
    top_books, sim_scores = recommandation_content_top_k(title, embeddings, model, content_df, knn=knn, k=top_k)

    logger.info(f"Top {top_k} recommandations :")
    logger.info(f"\n{top_books[['title', 'authors']]}")

if __name__ == "__main__":
    app()

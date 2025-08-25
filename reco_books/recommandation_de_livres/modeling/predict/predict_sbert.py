from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sentence_transformers import SentenceTransformer
import joblib

from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.content_utils import recommandation_content_top_k
from recommandation_de_livres.iads.utils import choose_dataset_interactively

app = typer.Typer()

DIR = choose_dataset_interactively()
print(f"Dataset choisi : {DIR}")

# --- Choix simple pour KNN ---
use_knn_input = input("Voulez-vous utiliser KNN pour accélérer la recherche ? (o/n) : ").strip().lower() or "o"
use_knn = use_knn_input == "o"
if use_knn:
    knn_path = MODELS_DIR / DIR / "knn_model_sbert.joblib"
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
    top_books, sim_scores = recommandation_content_top_k(title, embeddings, model, content_df, knn=knn, k=top_k)

    logger.info(f"Top {top_k} recommandations :")
    logger.info(f"\n{top_books[['title', 'authors']]}")


if __name__ == "__main__":
    app()
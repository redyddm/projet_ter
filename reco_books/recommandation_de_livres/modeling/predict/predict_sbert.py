import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sentence_transformers import SentenceTransformer
import joblib

from sklearn.metrics.pairwise import cosine_similarity

from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.content_utils import (
    recommandation_content_top_k,
    user_profile_embedding,
    recommandation_content_user_top_k
)
from recommandation_de_livres.iads.utils import choose_dataset_interactively

app = typer.Typer()

# --- Choix dataset ---
DIR = choose_dataset_interactively()
print(f"Dataset choisi : {DIR}")

# --- Choix simple pour KNN ---
use_knn_input = input("Voulez-vous utiliser KNN pour accélérer la recherche ? (o/n) [o] : ").strip().lower() or "o"
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

# --- Chemins modèles / données ---
ratings_path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet"
model_sbert_path = MODELS_DIR / DIR / "sbert_model"
embeddings_sbert_path = PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy"
content_path = PROCESSED_DATA_DIR / DIR / "content_dataset.parquet"

@app.command()
def main(top_k: int = typer.Option(5, prompt="Nombre de recommandations souhaité")):
    logger.info("Chargement du contenu et des modèles...")
    content_df = load_parquet(content_path)
    model = SentenceTransformer(str(model_sbert_path))
    embeddings = np.load(embeddings_sbert_path)
    ratings = load_parquet(ratings_path)

    # --- Choix interactif ---
    choice = input("Souhaitez-vous générer des recommandations basées sur un profil utilisateur ou un titre ? [profil/titre] : ").strip().lower()
    
    if choice == "profil":
        user_id = int(input("Entrez l'ID de l'utilisateur : ").strip())
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(content_df['item_id'])}
        user_vec = user_profile_embedding(user_id, ratings, embeddings, item_id_to_idx)

        if user_vec is None:
            print(f"L'utilisateur {user_id} n'a pas encore de notes. Veuillez fournir un titre pour cold-start.")
            title = input("Titre du livre pour la recommandation : ").strip()
            top_books, sim_scores = recommandation_content_top_k(title, embeddings, model, content_df, knn=knn, k=top_k)
        else:
            top_books, sim_scores = recommandation_content_user_top_k(user_id, embeddings, model, content_df, ratings, knn=knn, k=top_k)

    elif choice == "titre":
        title = input("Titre du livre pour la recommandation : ").strip()
        top_books, sim_scores = recommandation_content_top_k(title, embeddings, model, content_df, knn=knn, k=top_k)

    else:
        print("Choix invalide. Veuillez choisir 'profil' ou 'titre'.")
        return

    # --- Affichage ---
    logger.info(f"\nTop-{top_k} recommandations :")
    logger.info(f"\n{top_books[['title', 'authors']]}")

if __name__ == "__main__":
    app()

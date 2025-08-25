from pathlib import Path
from loguru import logger
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.iads.utils import choose_dataset_interactively

# Choix interactif du dataset
DIR = choose_dataset_interactively()
print(f"Dataset choisi : {DIR}")

# Choix simple du type d'embeddings
print("Quel type d'embeddings utiliser ?")
print("1 - Word2Vec")
print("2 - SBERT")
choice = input("Entrez 1 ou 2 : ").strip()

if choice == "1":
    embeddings_path = PROCESSED_DATA_DIR / DIR / "embeddings_w2v.npy"
    knn_model_path = MODELS_DIR / DIR / "knn_model_w2v.joblib"
elif choice == "2":
    embeddings_path = PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy"
    knn_model_path = MODELS_DIR / DIR / "knn_model_sbert.joblib"
else:
    raise ValueError("Choix invalide. Entrez 1 ou 2.")

print(f"Embeddings choisis : {embeddings_path}")

# Paramètres du KNN
n_neighbors = 11
metric = "cosine"

# ---- Chargement des embeddings ----
logger.info(f"Chargement des embeddings depuis {embeddings_path}")
embeddings = np.load(embeddings_path)
logger.info(f"Dimensions des embeddings : {embeddings.shape}")

# ---- Entraînement du modèle KNN ----
logger.info("Initialisation et entraînement du modèle KNN...")
knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
knn.fit(embeddings)

# ---- Sauvegarde du modèle ----
knn_model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(knn, knn_model_path)
logger.success(f"Modèle KNN sauvegardé dans {knn_model_path}")
logger.success("Entraînement KNN terminé avec succès.")

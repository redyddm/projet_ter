from pathlib import Path
from sentence_transformers import SentenceTransformer

from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import typer
import pickle
import torch

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
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / DIR / "features_sbert.pkl",
    model_path: Path = MODELS_DIR / DIR / "sbert_model",
    embeddings_path = PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy",
    # -----------------------------------------
):
    logger.info("Loading the features...")

    content_df = pd.read_pickle(features_path)

    logger.info("Loading a pretrained Sentence-Bert model...")

    sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Creating Sentence-Bert embeddings...")

    embeddings=sbert.encode(content_df['text_clean'], convert_to_numpy=True, batch_size=64, show_progress_bar=True)
    
    logger.success("Embeddings creation complete.")

    logger.info(f"Saving the Sentence-BERT model to {model_path}")
    
    sbert.save(str(model_path))

    logger.success("Sentence-BERT model saved.")

    logger.info(f"Saving embeddings to {embeddings_path}")

    np.save(embeddings_path, embeddings)

    logger.success("Embeddings saved")

if __name__ == "__main__":
    app()

from surprise import NMF, Dataset, Reader
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import pickle

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_parquet

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
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet",
    model_path: Path = MODELS_DIR / DIR / "nmf_model.pkl",
    # -----------------------------------------
):
    logger.info("Loading the features...")

    collaborative_df = load_parquet(features_path)

    if choice == "1":
        reader = Reader(rating_scale=(1,10))

    elif choice == "2":
        reader = Reader(rating_scale=(1,5))

    data = Dataset.load_from_df(collaborative_df[['user_id','item_id','rating']], reader)
    trainset = data.build_full_trainset()

    logger.info("Creating a NMF model...")

    n_factors = input("Nombre de facteurs latents pour NMF [50] : ")
    n_factors = int(n_factors) if n_factors.strip() != "" else 50

    n_epochs = input("Nombre d'epochs [50] : ")
    n_epochs = int(n_epochs) if n_epochs.strip() != "" else 50

    reg_pu = input("Régularisation des utilisateurs [0.06] : ")
    reg_pu = float(reg_pu) if reg_pu.strip() != "" else 0.06

    reg_qi = input("Régularisation des livres [0.06] : ")
    reg_qi = float(reg_pu) if reg_qi.strip() != "" else 0.06

    nmf = NMF(n_factors=n_factors, n_epochs=n_epochs, reg_qi=reg_qi, reg_pu=reg_pu)

    logger.info("Training the NMF model...")
    
    nmf.fit(trainset)

    logger.success("Modeling training complete.")

    logger.success(f"Saving the NMF model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(nmf, f)

if __name__ == "__main__":
    app()

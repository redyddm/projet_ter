from surprise import SVD, Dataset, Reader
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import pickle

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.iads.utils import choose_dataset_interactively
from recommandation_de_livres.loaders.load_data import load_parquet

app = typer.Typer()

DIR = choose_dataset_interactively()
print(f"Dataset choisi : {DIR}")

raing_max_input = input("Note maximale possible [5] : ")
rating_max = int(raing_max_input) if raing_max_input.strip() != "" else 5

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet",
    model_path: Path = MODELS_DIR / DIR / "svd_model.pkl",
    # -----------------------------------------
):
    logger.info("Loading the features...")

    collaborative_df = load_parquet(features_path)

    reader = Reader(rating_scale=(1, rating_max))

    data = Dataset.load_from_df(collaborative_df[['user_id','item_id','rating']], reader)
    trainset = data.build_full_trainset()

    logger.info("Creating a SVD model...")

    n_factors = input("Nombre de facteurs latents pour SVD [50] : ")
    n_factors = int(n_factors) if n_factors.strip() != "" else 50

    n_epochs = input("Nombre d'epochs [50] : ")
    n_epochs = int(n_epochs) if n_epochs.strip() != "" else 50

    lr_all = input("Learning rate [0.002] : ")
    lr_all = float(lr_all) if lr_all.strip() != "" else 0.002

    reg_all = input("Régularisation [0.02] : ")
    reg_all = float(reg_all) if reg_all.strip() != "" else 0.02

    svd = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

    logger.info("Training the SVD model...")
    
    svd.fit(trainset)

    logger.success("Modeling training complete.")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.success(f"Saving the SVD model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(svd, f)

if __name__ == "__main__":
    app()
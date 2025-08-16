from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import pickle
import gensim

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features_w2v.pkl",
    model_path: Path = MODELS_DIR / "word2vec_model.pkl",
    # -----------------------------------------
):
    logger.info("Loading the features...")

    content_df = pd.read_pickle(features_path)

    logger.info("Creating a Word2Vec model...")

    w2v=gensim.models.Word2Vec(sentences=text_fusion, vector_size=150, window=5, workers=14, min_count=2)


    logger.info("Training some model...")
    
    svd.fit(trainset)

    logger.success("Modeling training complete.")

    logger.success(f"Saving the SVD model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(svd, f)

if __name__ == "__main__":
    app()

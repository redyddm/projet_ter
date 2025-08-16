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
    model_path: Path = MODELS_DIR / "word2vec.model",
    # -----------------------------------------
):
    logger.info("Loading the features...")

    content_df = pd.read_pickle(features_path)

    logger.info("Creating a Word2Vec model...")

    w2v=gensim.models.Word2Vec(vector_size=150, window=5, workers=10, min_count=2)

    logger.info("Building the vocabulary...")

    w2v.build_vocab(content_df['text_clean'])

    logger.info("Training the Word2Vec model...")

    w2v.train(content_df['text_clean'], epochs=w2v.epochs, total_examples=w2v.corpus_count)
    
    logger.success("Modeling training complete.")

    logger.info(f"Saving the Word2Vec model to {model_path}")
    
    w2v.save(str(model_path))

    logger.success("Word2Vec model saved.")

if __name__ == "__main__":
    app()

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import typer
import gensim
import multiprocessing as mp

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.iads.content_utils import get_text_vector
from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.iads.progress_w2v import TqdmCorpus, EpochLogger

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
    features_path: Path = PROCESSED_DATA_DIR / DIR / "features_w2v.parquet",
    model_path: Path = MODELS_DIR / DIR / "word2vec.model",
    embeddings_path: Path = PROCESSED_DATA_DIR / DIR / "embeddings_w2v.npy",
    vector_size: int = 300,
    window: int = 10,
    min_count: int = 2,
    epochs: int = 5,
    # -----------------------------------------
):
    logger.info("Loading the features...")

    content_df = load_parquet(features_path)

    logger.info("Creating a Word2Vec model...")

    corpus = TqdmCorpus(content_df['text_clean'].apply(lambda x: list(x)))

    w2v = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=mp.cpu_count(),
        sg=0
    )

    logger.info("Building the vocabulary...")

    w2v.build_vocab(corpus)

    logger.info("Training the Word2Vec model...")

    w2v.train(
        corpus,
        total_examples=w2v.corpus_count,
        epochs=epochs,
        callbacks=[EpochLogger()]
    )

    logger.success("Model training complete.")

    logger.info(f"Saving the Word2Vec model to {model_path}")
    w2v.save(str(model_path))
    logger.success("Word2Vec model saved.")

    logger.info("Calculating the embeddings...")

    book_embeddings = np.vstack([
        get_text_vector(tokens, w2v) for tokens in tqdm(corpus, desc="Calcul des embeddings")
    ])

    logger.info(f"Saving the embeddings to {embeddings_path}")
    np.save(embeddings_path, book_embeddings)
    logger.success("Embeddings saved.")

if __name__ == "__main__":
    app()
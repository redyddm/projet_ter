from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import typer
import gensim

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.iads.content_utils import get_text_vector
from recommandation_de_livres.iads.progress_w2v import TqdmCorpus, EpochLogger

app = typer.Typer()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features_w2v.pkl",
    model_path: Path = MODELS_DIR / "word2vec.model",
    embeddings_path: Path = PROCESSED_DATA_DIR / "embeddings_w2v.npy",
    vector_size: int = 300,
    window: int = 10,
    min_count: int = 2,
    epochs: int = 5,
    # -----------------------------------------
):
    logger.info("Loading the features...")

    content_df = pd.read_pickle(features_path)

    logger.info("Creating a Word2Vec model...")

    corpus = TqdmCorpus(content_df['text_clean'])

    w2v = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=10,
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
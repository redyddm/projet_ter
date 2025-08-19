from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import gensim
import typer

from recommandation_de_livres.config import PROCESSED_DATA_DIR
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle
from recommandation_de_livres.iads.text_cleaning import nettoyage_texte, nettoyage_avance

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "content_dataset.pkl",
    output_path_csv: Path = PROCESSED_DATA_DIR / "features_tfidf.csv",
    output_path_pkl: Path = PROCESSED_DATA_DIR / "features_tfidf.pkl",
    # -----------------------------------------
):
    logger.info('Loading content dataset...')

    content_df = pd.read_pickle(input_path)

    logger.info('Fusionning the texts for TF-IDF...')

    content_df['text_for_tfidf'] = (
        content_df['title'].fillna('') + " " +
        content_df['authors'].fillna('')
    )

    logger.info("Cleaning and tokenizing the text...")

    tqdm.pandas(desc='Nettoyage des textes')
    content_df['text_clean'] = content_df['text_for_tfidf'].progress_apply(nettoyage_avance)

    logger.info("Creating the features dataframe...")

    features_df = pd.DataFrame({
        'isbn': content_df['isbn'],
        'text_clean': content_df['text_clean']
    })

    logger.info("Saving the features...")

    save_df_to_csv(features_df, output_path_csv)
    save_df_to_pickle(features_df, output_path_pkl)

    logger.success("TF-IDF features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import gensim
import typer

from recommandation_de_livres.config import PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.iads.text_cleaning import nettoyage_texte, nettoyage_avance

app = typer.Typer()

choice = input("Choix du dataset [2] : Recommender (1), Goodreads (2), Fusion (3) ") or "2"

if choice == "1":
    DIR = "recommender"
elif choice == "2":
    DIR = "goodreads"
elif choice == "3":
    DIR = "fusion"
else:
    raise ValueError("Choix invalide (1, 2 ou 3 attendu)")

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / DIR / "content_dataset.parquet",
    output_path_csv: Path = PROCESSED_DATA_DIR / DIR / "features_tfidf.csv",
    output_path_parquet: Path = PROCESSED_DATA_DIR / DIR / "features_tfidf.parquet",
    # -----------------------------------------
):
    logger.info('Loading content dataset...')

    content_df = load_parquet(input_path)

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
    'item_id': content_df['item_id'],
    'text_clean': content_df['text_clean']
    })

    logger.info("Saving the features...")

    output_path_csv.parent.mkdir(parents=True, exist_ok=True)
    output_path_parquet.parent.mkdir(parents=True, exist_ok=True)

    save_df_to_csv(features_df, output_path_csv)
    save_df_to_parquet(features_df, output_path_parquet)

    logger.success("TF-IDF features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import gensim
import typer

from recommandation_de_livres.config import PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.iads.content_utils import combine_text
from recommandation_de_livres.iads.text_cleaning import nettoyage_texte

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
    input_path: Path = PROCESSED_DATA_DIR / DIR / "content_dataset.parquet",
    output_path_csv: Path = PROCESSED_DATA_DIR / DIR / "features_w2v.csv",
    output_path_parquet: Path = PROCESSED_DATA_DIR / DIR / "features_w2v.parquet",
    # -----------------------------------------
):
    logger.info('Loading content dataset...')

    content_df = load_parquet(input_path)
    
    logger.info("Cleaning and tokenizing the text...")

    tqdm.pandas(desc='Nettoyage des textes')

    content_df['text_combined'] = content_df.progress_apply(lambda row: combine_text(row, choice), axis=1)
    content_df['text_clean'] = content_df['text_combined'].progress_apply(gensim.utils.simple_preprocess)

    logger.info("Creating the features dataframe...")
    
    if choice =="3":
        features_df = pd.DataFrame({
        'book_id': content_df['book_id'],
        'text_clean': content_df['text_clean']
        })

    else :
        features_df = pd.DataFrame({
            'isbn': content_df['isbn'],
            'text_clean': content_df['text_clean']
        })

    logger.info("Saving the features...")

    save_df_to_csv(features_df, output_path_csv)
    save_df_to_parquet(features_df, output_path_parquet)

    logger.success("Word2Vec features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
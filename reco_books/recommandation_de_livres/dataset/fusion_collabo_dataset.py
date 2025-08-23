from pathlib import Path
from loguru import logger
import typer
import pandas as pd

from recommandation_de_livres.loaders import load_data
from recommandation_de_livres.iads.collabo_utils import rescale_ratings
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.iads.create_users import create_users_file
from recommandation_de_livres.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    ratings_path_recommender: Path = PROCESSED_DATA_DIR / "recommender" / "collaborative_dataset.parquet",
    ratings_path_goodreads: Path = PROCESSED_DATA_DIR / "goodreads" / "collaborative_dataset.parquet",
    output_path_csv: Path = PROCESSED_DATA_DIR / "fusion" / "collaborative_dataset.csv",
    output_path_parquet: Path = PROCESSED_DATA_DIR / "fusion" / "collaborative_dataset.parquet",
    output_users: Path =PROCESSED_DATA_DIR / "fusion" / "users.csv",
):
    logger.info("Chargement des datasets processed...")

    ratings_recommender = load_data.load_parquet(ratings_path_recommender)
    ratings_goodreads = load_data.load_parquet(ratings_path_goodreads)

    # Harmoniser les colonnes (intersection)
    common_cols = list(set(ratings_recommender.columns) & set(ratings_goodreads.columns))
    ratings_recommender = ratings_recommender[common_cols]
    ratings_goodreads = ratings_goodreads[common_cols]

    ratings_recommender['rating'] = rescale_ratings(ratings_recommender['rating'], 1, 5)
    ratings_goodreads['rating'] = rescale_ratings(ratings_goodreads['rating'], 1, 5)

    # Fusion & suppression doublons ISBN si présent
    ratings = pd.concat([ratings_recommender, ratings_goodreads], ignore_index=True)

    ratings['user_id']=ratings['user_id'].astype(str)
    ratings['item_id']=ratings['item_id'].astype(str)

    cats = ratings['user_id'].astype("category")
    ratings['user_index'] = cats.cat.codes + 1  

    cats_item = ratings['item_id'].astype("category")
    ratings['item_index'] = cats_item.cat.codes + 1 


    logger.info(f"Fusion terminée : {len(ratings)} notes au total.")

    # Sauvegarde
    output_path_csv.parent.mkdir(parents=True, exist_ok=True)
    output_path_parquet.parent.mkdir(parents=True, exist_ok=True)
    create_users_file(output_path_csv, output_users)
    save_df_to_csv(ratings, output_path_csv)
    save_df_to_parquet(ratings, output_path_parquet)

    logger.success(f"Dataset fusionné sauvegardé dans : {output_path_parquet}")

if __name__ == "__main__":
    app()

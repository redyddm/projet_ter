from pathlib import Path
from loguru import logger
import typer
import pickle

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.iads.utils import choose_dataset_interactively
from recommandation_de_livres.loaders.load_data import load_parquet

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS

app = typer.Typer()

DIR = choose_dataset_interactively()
print(f"Dataset choisi : {DIR}")

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet",
    model_path: Path = MODELS_DIR / DIR / "als_model.pkl",
):
    # Initialiser Spark
    spark = SparkSession.builder.appName("ALS_Training").getOrCreate()

    logger.info("Loading the collaborative dataset...")
    collaborative_df = load_parquet(features_path)

    # Convertir Pandas -> Spark DataFrame
    df = spark.createDataFrame(collaborative_df)

    # Vérifier si item_id est numérique

    if dict(df.dtypes)["item_id"] != "int" and dict(df.dtypes)["item_id"] != "bigint":
        logger.info("Indexing item_id...")
        item_indexer = StringIndexer(inputCol="item_id", outputCol="itemIndex").fit(df)
        df = item_indexer.transform(df)
    else:
        df = df.withColumnRenamed("item_id", "itemIndex")

    df = df.select("user_index", "itemIndex", "rating")

    # Paramètres ALS interactifs
    rank = input("Nombre de facteurs latents [50] : ")
    rank = int(rank) if rank.strip() != "" else 50

    maxIter = input("Nombre d'epochs [20] : ")
    maxIter = int(maxIter) if maxIter.strip() != "" else 20

    regParam = input("Régularisation [0.1] : ")
    regParam = float(regParam) if regParam.strip() != "" else 0.1

    implicit = input("Feedback implicite ? (y/N) : ")
    implicit = True if implicit.lower() == "y" else False

    logger.info("Training ALS model...")
    als = ALS(
        userCol="userIndex",
        itemCol="itemIndex",
        ratingCol="rating",
        rank=rank,
        maxIter=maxIter,
        regParam=regParam,
        implicitPrefs=implicit,
        coldStartStrategy="drop"
    )

    model = als.fit(df)

    logger.success("ALS model trained.")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.success(f"Saving ALS model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    spark.stop()

if __name__ == "__main__":
    app()

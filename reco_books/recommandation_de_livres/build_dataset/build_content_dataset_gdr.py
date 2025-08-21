from recommandation_de_livres.preprocessing.preprocess_content_gdr import *
from recommandation_de_livres.config import INTERIM_DATA_DIR
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet

def build_content_dataset(books, authors, categories):

    # Mapping IDs → noms et conversion en string
    books = map_author_names(books, authors)

    save_df_to_csv(books, INTERIM_DATA_DIR / 'goodreads' / "books_authors.csv")
    save_df_to_parquet(books, INTERIM_DATA_DIR / 'goodreads' / "books_authors.parquet")

    # Ajout de langue pour les lignes où on ne sait pas
    books=add_language_column(books)

    # Ajout des catégories maintenant
    books = add_categories_columns(books, categories)

    # Filtrage de base (NaN, doublons, langue)
    books = filter_books_basic(books)

    # Suppression des titres en doublon
    books = remove_duplicates(books)

    return books

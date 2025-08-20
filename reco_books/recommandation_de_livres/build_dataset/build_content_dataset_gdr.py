from recommandation_de_livres.preprocessing.preprocess_content_gdr import *
from recommandation_de_livres.config import INTERIM_DATA_DIR

def build_content_dataset(books, authors, categories):

    # Mapping IDs → noms et conversion en string
    books = map_author_names(books, authors)

    books.to_csv(INTERIM_DATA_DIR / 'goodreads' / "books_authors.csv")

    # Ajout de langue pour les lignes où on ne sait pas
    books=add_language_column(books)

    # Ajout des catégories maintenant
    books = add_categories_columns(books, categories)

    # Filtrage de base (NaN, doublons, langue)
    books = filter_books_basic(books)

    # Suppression des titres en doublon
    books = remove_duplicates(books)

    return books

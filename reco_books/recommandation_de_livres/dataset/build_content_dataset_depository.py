from recommandation_de_livres.preprocessing.preprocess_content_depository import *

def build_content_dataset(books, authors, categories):
    # Sélection et renommage des colonnes
    books = select_and_rename_books_columns(books)

    # Filtrage de base (NaN, doublons, langue)
    books = filter_books_basic(books)

    # Mapping IDs → noms et conversion en string
    books = map_ids_to_names(books, authors, categories)

    # Suppression des titres en doublon
    books = remove_duplicates(books)

    return books

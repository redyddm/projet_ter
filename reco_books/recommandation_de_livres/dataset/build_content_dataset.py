from recommandation_de_livres.preprocessing.preprocess_content import *

def build_content_dataset(books):

    books.dropna(inplace=True)

    # Sélection et renommage des colonnes
    books = select_and_rename_books_columns(books)

    # Récupération des descriptions si possible via openlibrary
    books = get_descriptions(books, update=False)

    # Filtrage de base (NaN)
    books = filter_books_basic(books)

    # Suppression des titres en doublon
    books = remove_duplicates(books)

    books['description']=books['description'].apply(list_to_str)

    return books

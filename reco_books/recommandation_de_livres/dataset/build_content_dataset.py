from recommandation_de_livres.preprocessing.preprocess_content import *
from recommandation_de_livres.config import INTERIM_DATA_DIR
import pandas as pd
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle

def build_content_dataset(books, update=False):

    books.dropna(inplace=True)

    # Sélection et renommage des colonnes
    books = select_and_rename_books_columns(books)

    # Récupération des descriptions si possible via openlibrary
    if update : #la première fois
        books = get_descriptions(books)
        save_df_to_csv(books, INTERIM_DATA_DIR / "books_desc.csv")
        save_df_to_pickle(books, INTERIM_DATA_DIR / "books_desc.pkl")
        
    else :
        books = pd.read_pickle(INTERIM_DATA_DIR / "books_desc.pkl")

    # Ajout de la langue pour éviter les doublons internationaux
    books = add_language_column(books)

    # Filtrage de base (NaN)
    books = filter_books_basic(books)

    # Suppression des titres en doublon
    books = remove_duplicates(books)

    books['description']=books['description'].apply(list_to_str)

    return books

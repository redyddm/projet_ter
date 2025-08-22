# preprocess_uniform.py
import pandas as pd
from pathlib import Path
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle

def rename_ratings_columns(ratings, user_col="User-ID", item_col=None, rating_col="Book-Rating"):
    """
    Renomme les colonnes pour uniformiser les datasets utilisateurs.
    Si item_col est None, sera défini plus tard via unify_book_ids.
    """
    ratings = ratings.copy()
    ratings.rename(columns={user_col: "user_id", rating_col: "rating"}, inplace=True)
    if item_col is not None:
        ratings.rename(columns={item_col: "item_id"}, inplace=True)
    return ratings

def unify_book_ids(df):
    """
    Définit la colonne 'item_id' à partir de 'book_id' ou 'ISBN' si item_id n'existe pas.
    """
    df = df.copy()
    if 'item_id' not in df.columns:
        if 'book_id' in df.columns:
            df['item_id'] = df['book_id']
        elif 'ISBN' in df.columns:
            df['item_id'] = df['ISBN']
        else:
            raise ValueError("Le dataset n'a ni 'book_id' ni 'ISBN'.")
    return df

def rename_books_columns(books,
                         isbn_col=None,
                         book_id_col=None,
                         language_col=None,
                         title_col=None,
                         author_col=None,
                         publisher_col=None,
                         year_col=None,
                         image_col=None):
    """
    Renomme les colonnes d'un dataset de livres pour uniformiser.
    Les colonnes peuvent être None si elles n'existent pas.
    Retourne un dataframe avec les noms standards :
    'item_id', 'title', 'authors', 'publisher', 'year', 'image_url'
    """
    books = books.copy()
    
    # Définir item_id à partir de book_id ou ISBN
    if book_id_col is not None and book_id_col in books.columns:
        books['item_id'] = books[book_id_col]
    elif isbn_col is not None and isbn_col in books.columns:
        books['item_id'] = books[isbn_col]

    # Mapping des autres colonnes
    rename_map = {}
    if title_col is not None: rename_map[title_col] = "title"
    if author_col is not None: rename_map[author_col] = "authors"
    if language_col is not None: rename_map[language_col] = "language"
    if publisher_col is not None: rename_map[publisher_col] = "publisher"
    if year_col is not None: rename_map[year_col] = "year"
    if image_col is not None: rename_map[image_col] = "image_url"

    books.rename(columns=rename_map, inplace=True)
    
    return books

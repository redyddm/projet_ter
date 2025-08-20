import pandas as pd
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle

def preprocess_collaborative(ratings, path):

    # Trier + reset index
    ratings.sort_values(by='user_id', inplace=True)
    ratings.reset_index(drop=True, inplace=True)

    # Filtrage par nombre moyen d’interactions utilisateur
    #moy_nb_inter = ratings_explicit['user_id'].value_counts().mean()
    mask_rating = ratings['user_id'].value_counts() > 50
    user_indexes = mask_rating[mask_rating].index

    ratings_tmp = ratings[ratings['user_id'].isin(user_indexes)].reset_index(drop=True).copy()

    # Filtrage par nombre moyen d’interactions livre
    #moy_books_inter = ratings_tmp['ISBN'].value_counts().mean()
    mask_books = ratings_tmp['book_id'].value_counts() > 50
    book_indexes = mask_books[mask_books].index

    ratings_explicit_filtered = ratings_tmp[ratings_tmp['book_id'].isin(book_indexes)].reset_index(drop=True).copy()

    # Sauvegarde des datasets filtrés
    save_df_to_csv(ratings_explicit_filtered, path / 'ratings_explicit_filtered.csv')
    save_df_to_pickle(ratings_explicit_filtered, path / 'ratings_explicit_filtered.pkl')

    return ratings_explicit_filtered

def add_book_metadata(ratings_tmp, books):

    books_subset = books[['book_id', 'isbn', 'title', 'authors', 'publisher', 'description','image_url']]

    ratings_tmp_books = ratings_tmp.merge(books_subset, on='book_id')

    # Supprimer les doublons user-livre
    ratings_tmp_books = ratings_tmp_books.drop_duplicates(subset=['user_id', 'book_id'])

    return ratings_tmp_books

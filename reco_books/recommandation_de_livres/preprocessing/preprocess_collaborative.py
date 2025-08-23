import pandas as pd
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle

def preprocess_collaborative(ratings, min_rating_book = 10, min_rating_user = 10):
    
    # Séparer explicite / implicite
    ratings_explicit = ratings[ratings['rating'] != 0].copy()
    ratings_implicit = ratings[ratings['rating'] == 0].copy()

    #save_df_to_csv(ratings_implicit, path / 'ratings_explicit.csv')
    #save_df_to_pickle(ratings_implicit, path / 'ratings_explicit.pkl')

    #save_df_to_csv(ratings_implicit, path / 'ratings_implicit.csv')
    #save_df_to_pickle(ratings_implicit, path / 'ratings_implicit.pkl')

    # Trier + reset index
    ratings_explicit.sort_values(by='user_id', inplace=True)
    ratings_explicit.reset_index(drop=True, inplace=True)

    ratings_implicit.sort_values(by='user_id', inplace=True)
    ratings_implicit.reset_index(drop=True, inplace=True)

    # Filtrage par nombre moyen d’interactions utilisateur
    #moy_nb_inter = ratings_explicit['user_id'].value_counts().mean()
    mask_rating = ratings_explicit['user_id'].value_counts() > min_rating_user
    user_indexes = mask_rating[mask_rating].index

    ratings_tmp = ratings_explicit[ratings_explicit['user_id'].isin(user_indexes)].reset_index(drop=True).copy()

    # Filtrage par nombre moyen d’interactions livre
    #moy_books_inter = ratings_tmp['item_id'].value_counts().mean()
    mask_books = ratings_tmp['item_id'].value_counts() > min_rating_book
    book_indexes = mask_books[mask_books].index

    ratings_explicit_filtered = ratings_tmp[ratings_tmp['item_id'].isin(book_indexes)].reset_index(drop=True).copy()

    # Filtrage de ratings implicit pour garder le datset cohérent
    ratings_implicit_filtered = ratings_implicit[
    (ratings_implicit['user_id'].isin(ratings_tmp['user_id'])) &
    (ratings_implicit['item_id'].isin(ratings_tmp['item_id']))
    ].reset_index(drop=True).copy()

    # Sauvegarde des datasets filtrés
    #save_df_to_csv(ratings_explicit_filtered, path / 'ratings_explicit_filtered.csv')
    #save_df_to_pickle(ratings_explicit_filtered, path / 'ratings_explicit_filtered.pkl')

    #save_df_to_csv(ratings_implicit_filtered, path / 'ratings_implicit_filtered.csv')
    #save_df_to_pickle(ratings_implicit_filtered, path / 'ratings_implicit_filtered.pkl')

    return ratings_explicit_filtered, ratings_implicit_filtered


def add_book_metadata(ratings_tmp, books):

    books_subset = books[['item_id', 'isbn', 'title', 'authors', 'publisher', 'image_url']]

    ratings_tmp_books = ratings_tmp.merge(books_subset, on='item_id')

    # Supprimer les doublons user-livre
    ratings_tmp_books = ratings_tmp_books.drop_duplicates(subset=['user_id', 'item_id'])

    return ratings_tmp_books

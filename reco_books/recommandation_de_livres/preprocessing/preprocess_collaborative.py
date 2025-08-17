import pandas as pd
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle
from recommandation_de_livres.loaders.load_collaborative import load_books

def rename_ratings_columns(ratings):
    ratings = ratings.copy()
    ratings.rename(columns={
        'User-ID': 'user_id',
        'Book-Rating': 'rating'
    }, inplace=True)

    return ratings

def preprocess_collaborative(ratings, path):
    
    # Séparer explicite / implicite
    ratings_explicit = ratings[ratings['rating'] != 0].copy()
    ratings_implicit = ratings[ratings['rating'] == 0].copy()


    save_df_to_csv(ratings_implicit, path / 'ratings_explicit.csv')
    save_df_to_pickle(ratings_implicit, path / 'ratings_explicit.pkl')

    save_df_to_csv(ratings_implicit, path / 'ratings_implicit.csv')
    save_df_to_pickle(ratings_implicit, path / 'ratings_implicit.pkl')

    # Trier + reset index
    ratings_explicit.sort_values(by='user_id', inplace=True)
    ratings_explicit.reset_index(drop=True, inplace=True)

    ratings_implicit.sort_values(by='user_id', inplace=True)
    ratings_implicit.reset_index(drop=True, inplace=True)

    # Filtrage par nombre moyen d’interactions utilisateur
    moy_nb_inter = ratings_explicit['user_id'].value_counts().mean()
    mask_rating = ratings_explicit['user_id'].value_counts() > moy_nb_inter
    user_indexes = mask_rating[mask_rating].index

    ratings_tmp = ratings_explicit[ratings_explicit['user_id'].isin(user_indexes)].reset_index(drop=True).copy()

    # Filtrage par nombre moyen d’interactions livre
    moy_books_inter = ratings_tmp['ISBN'].value_counts().mean()
    mask_books = ratings_tmp['ISBN'].value_counts() > moy_books_inter
    book_indexes = mask_books[mask_books].index

    ratings_explicit_filtered = ratings_tmp[ratings_tmp['ISBN'].isin(book_indexes)].reset_index(drop=True).copy()

    # Filtrage de ratings implicit pour garder le datset cohérent
    ratings_implicit_filtered = ratings_implicit[
    (ratings_implicit['user_id'].isin(ratings_tmp['user_id'])) &
    (ratings_implicit['ISBN'].isin(ratings_tmp['ISBN']))
    ].reset_index(drop=True).copy()

    # Sauvegarde des datasets filtrés
    save_df_to_csv(ratings_explicit_filtered, path / 'ratings_explicit_filtered.csv')
    save_df_to_pickle(ratings_explicit_filtered, path / 'ratings_explicit_filtered.pkl')

    save_df_to_csv(ratings_implicit_filtered, path / 'ratings_implicit_filtered.csv')
    save_df_to_pickle(ratings_implicit_filtered, path / 'ratings_implicit_filtered.pkl')

    return ratings_explicit_filtered, ratings_implicit_filtered

def preprocess_collaborative_diverse(ratings, path, min_user_ratings=1, min_book_ratings=1):
    """
    Prépare le dataset pour la recommandation collaborative
    en gardant tous les utilisateurs et livres avec un minimum de notes,
    plutôt que de filtrer par moyenne.
    """
    ratings = ratings.copy()

    # Séparer explicite / implicite
    ratings_explicit = ratings[ratings['rating'] != 0].copy()
    ratings_implicit = ratings[ratings['rating'] == 0].copy()

    # Sauvegarde brute
    save_df_to_csv(ratings_explicit, path / 'ratings_explicit.csv')
    save_df_to_pickle(ratings_explicit, path / 'ratings_explicit.pkl')
    save_df_to_csv(ratings_implicit, path / 'ratings_implicit.csv')
    save_df_to_pickle(ratings_implicit, path / 'ratings_implicit.pkl')

    # Trier + reset index
    ratings_explicit.sort_values(by='user_id', inplace=True)
    ratings_explicit.reset_index(drop=True, inplace=True)

    ratings_implicit.sort_values(by='user_id', inplace=True)
    ratings_implicit.reset_index(drop=True, inplace=True)

    # Filtrage par nombre minimum d’interactions utilisateur
    user_counts = ratings_explicit['user_id'].value_counts()
    users_to_keep = user_counts[user_counts >= min_user_ratings].index
    ratings_filtered = ratings_explicit[ratings_explicit['user_id'].isin(users_to_keep)].copy()

    # Filtrage par nombre minimum d’interactions livre
    book_counts = ratings_filtered['ISBN'].value_counts()
    books_to_keep = book_counts[book_counts >= min_book_ratings].index
    ratings_explicit_filtered = ratings_filtered[ratings_filtered['ISBN'].isin(books_to_keep)].reset_index(drop=True)

    # Adapter le dataset implicite pour cohérence
    ratings_implicit_filtered = ratings_implicit[
        (ratings_implicit['user_id'].isin(ratings_explicit_filtered['user_id'])) &
        (ratings_implicit['ISBN'].isin(ratings_explicit_filtered['ISBN']))
    ].reset_index(drop=True)

    # Sauvegarde des datasets filtrés
    save_df_to_csv(ratings_explicit_filtered, path / 'ratings_explicit_filtered.csv')
    save_df_to_pickle(ratings_explicit_filtered, path / 'ratings_explicit_filtered.pkl')

    save_df_to_csv(ratings_implicit_filtered, path / 'ratings_implicit_filtered.csv')
    save_df_to_pickle(ratings_implicit_filtered, path / 'ratings_implicit_filtered.pkl')

    return ratings_explicit_filtered, ratings_implicit_filtered


def add_book_metadata(ratings_tmp, books):
    books.rename(
        columns={
            'Book-Title': 'title',
            'Book-Author': 'author',
            'Year-Of-Publication': 'year',
            'Publisher': 'publisher'
        },
        inplace=True
    )
    books_subset = books[['ISBN', 'title', 'author', 'publisher', 'Image-URL-L']]

    ratings_tmp_books = ratings_tmp.merge(books_subset, on='ISBN')

    # Supprimer les doublons user-livre
    ratings_tmp_books = ratings_tmp_books.drop_duplicates(subset=['user_id', 'title'])

    return ratings_tmp_books

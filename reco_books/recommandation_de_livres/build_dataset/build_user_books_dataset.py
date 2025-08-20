from recommandation_de_livres.preprocessing.preprocess_user_book import *
from recommandation_de_livres.config import INTERIM_DATA_DIR
import pandas as pd
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_pickle

def build_user_book_dataset(books, ratings, users):

    books = rename_columns_books(books)
    ratings = rename_columns_ratings(ratings)
    users = rename_columns_users(users)

    ratings_books = fusion_ratings_books(books, ratings)

    users_final = fusion_user_ratings(users, ratings_books)

    return users_final

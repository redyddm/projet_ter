from preprocessing.preprocess_collaborative import rename_ratings_columns, preprocess_collaborative, add_book_metadata
from config import INTERIM_DATA_DIR

def build_collaborative_dataset(books, ratings):

    ratings = rename_ratings_columns(ratings)

    ratings_explicit, _ = preprocess_collaborative(ratings, INTERIM_DATA_DIR)

    collaborative_dataset = add_book_metadata(ratings_explicit, books)

    return collaborative_dataset

from recommandation_de_livres.preprocessing.preprocess_collaborative import rename_ratings_columns, preprocess_collaborative, preprocess_collaborative_diverse, add_book_metadata
from recommandation_de_livres.config import INTERIM_DATA_DIR

def build_collaborative_dataset(books, ratings):

    ratings = rename_ratings_columns(ratings)

    ratings_explicit, _ = preprocess_collaborative(ratings, INTERIM_DATA_DIR)

    collaborative_dataset = add_book_metadata(ratings_explicit, books)

    return collaborative_dataset

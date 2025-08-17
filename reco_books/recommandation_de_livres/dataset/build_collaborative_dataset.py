from recommandation_de_livres.preprocessing.preprocess_collaborative import rename_ratings_columns, preprocess_collaborative, preprocess_collaborative_diverse, add_book_metadata
from recommandation_de_livres.config import INTERIM_DATA_DIR

def build_collaborative_dataset(books, ratings):

    ratings = rename_ratings_columns(ratings)

    # Ici on utilise la moyenne des interactions pour filtrer
    #ratings_explicit, _ = preprocess_collaborative(ratings, INTERIM_DATA_DIR)


    # Ici on choisit le minimum d'interaction pour filtrer
    ratings_explicit, _ = preprocess_collaborative_diverse(ratings, INTERIM_DATA_DIR, min_book_ratings=10, min_user_ratings=20)

    collaborative_dataset = add_book_metadata(ratings_explicit, books)

    return collaborative_dataset

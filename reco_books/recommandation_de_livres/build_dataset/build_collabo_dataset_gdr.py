from recommandation_de_livres.preprocessing.preprocess_collabo_gdr import preprocess_collaborative, add_book_metadata
from recommandation_de_livres.config import INTERIM_DATA_DIR

DIR = 'goodreads'

def build_collaborative_dataset(books, ratings):

    ratings_explicit = preprocess_collaborative(ratings, INTERIM_DATA_DIR / DIR)

    collaborative_dataset = add_book_metadata(ratings_explicit, books)

    return collaborative_dataset

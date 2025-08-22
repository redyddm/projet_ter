from recommandation_de_livres.preprocessing.preprocess_collaborative import preprocess_collaborative, add_book_metadata
from recommandation_de_livres.config import INTERIM_DATA_DIR

def build_collaborative_dataset(books, ratings):

    ratings_explicit, _ = preprocess_collaborative(ratings)

    collaborative_dataset = add_book_metadata(ratings_explicit, books)

    cats = collaborative_dataset['user_id'].astype("category")
    collaborative_dataset['user_index'] = cats.cat.codes + 1  

    return collaborative_dataset

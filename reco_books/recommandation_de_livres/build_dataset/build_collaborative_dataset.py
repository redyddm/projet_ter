from recommandation_de_livres.preprocessing.preprocess_collaborative import preprocess_collaborative, add_book_metadata
from recommandation_de_livres.preprocessing.preprocess_content import map_author_names

def build_collaborative_dataset(books, ratings, authors, min_ratings=0, min_users_interaction=0):

    ratings_explicit, _ = preprocess_collaborative(ratings, min_rating_book=min_ratings, min_rating_user=min_users_interaction)

    collaborative_dataset = add_book_metadata(ratings_explicit, books)

    if authors is not None:
        collaborative_dataset = map_author_names(collaborative_dataset, authors, authors_col='authors')

    cats = collaborative_dataset['user_id'].astype("category")
    collaborative_dataset['user_index'] = cats.cat.codes + 1  

    return collaborative_dataset

from recommandation_de_livres.preprocessing.preprocess_collaborative import preprocess_collaborative, add_book_metadata
from recommandation_de_livres.preprocessing.preprocess_content import map_author_names

def build_collaborative_dataset(books, ratings, authors=None, min_ratings=0, min_users_interaction=0):
    """ Fonction permettant de créer le dataset collaboratif. Un index est aussi créé en se basant sur les user_id.
        Args:
            books (pd.DataFrame) : DataFrame des livres
            ratings (pd.DataFrame) : DataFrame des notes
            authors (pd.DataFrame) : DataFrame des auteurs avec leur id et leur nom
            min_ratings (int) : nombre de notes minimal que le livre a reçu
            min_users_interaction (int) : nombre de notes minimal que l'utilisateur a donné
        Returns:
            collaborative_dataset (pd.DataFrame) : DataFrame du dataset collaboratif complet
    """

    ratings_explicit, _ = preprocess_collaborative(ratings, min_ratings=min_ratings, min_users_interaction=min_users_interaction)

    collaborative_dataset = add_book_metadata(ratings_explicit, books)

    if authors is not None:
        books=map_author_names(books, authors)

    cats = collaborative_dataset['user_id'].astype("category") # Création de l'index unique
    collaborative_dataset['user_index'] = cats.cat.codes + 1  

    return collaborative_dataset
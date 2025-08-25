def preprocess_collaborative(ratings, min_ratings = 10, min_users_interaction = 10):
    """ Fonction permettant de faire le prétraitement des données du dataset ratings.
        Args:
            ratings (pd.DataFrame) : DataFrame des notes
            min_rating_book (int) : nombre de notes minimal que le livre a reçu
            min_users_interaction (int) : nombre de notes minimal que l'utilisateur a donné
        
        Returns:
            ratings_explicit_filtered (pd.DataFrame) : DataFrame des interactions explicites filtré
            ratings_implicit_filtered (pd.DataFrame) : DataFrame des interactions implicites filtré
    """
    
    # On sépare le dataset pour garder que les interactions explicites
    ratings_explicit = ratings[ratings['rating'] != 0].copy()
    ratings_implicit = ratings[ratings['rating'] == 0].copy()

    # On les trie pour regrouper les notes d'un même utilisateur
    ratings_explicit.sort_values(by='user_id', inplace=True)
    ratings_explicit.reset_index(drop=True, inplace=True)

    ratings_implicit.sort_values(by='user_id', inplace=True)
    ratings_implicit.reset_index(drop=True, inplace=True)

    # Filtrage par le nombre d'interaction faite par un utilisateur
    mask_rating = ratings_explicit['user_id'].value_counts() > min_users_interaction
    user_indexes = mask_rating[mask_rating].index

    ratings_tmp = ratings_explicit[ratings_explicit['user_id'].isin(user_indexes)].reset_index(drop=True).copy()

    # Filtrage par le nombre d’interactions qu'un livre a reçu
    mask_books = ratings_tmp['item_id'].value_counts() > min_ratings
    book_indexes = mask_books[mask_books].index

    ratings_explicit_filtered = ratings_tmp[ratings_tmp['item_id'].isin(book_indexes)].reset_index(drop=True).copy()

    # Filtrage de ratings implicit pour garder le dataset cohérent
    ratings_implicit_filtered = ratings_implicit[
    (ratings_implicit['user_id'].isin(ratings_tmp['user_id'])) &
    (ratings_implicit['item_id'].isin(ratings_tmp['item_id']))
    ].reset_index(drop=True).copy()

    return ratings_explicit_filtered, ratings_implicit_filtered


def add_book_metadata(ratings, books):
    """ Fonction ajoutant les données des livres du dataset book.
        Args:
            ratings (pd.DataFrame) : DataFrame des notes
            books (pd.DataFrame) : DataFrame des livres
        Returns:
            ratings_books (pd.DataFrame) : DataFrame des notes avec les données des livres

    """

    books_subset = books[['item_id', 'isbn', 'title', 'authors', 'publisher', 'image_url']]

    ratings_books = ratings.merge(books_subset, on='item_id')

    # Supprimer les doublons user-livre
    ratings_books = ratings_books.drop_duplicates(subset=['user_id', 'item_id'])

    return ratings_books

def rename_ratings_columns(ratings, user_col, rating_col, item_col=None):
    """
    Renomme les colonnes pour uniformiser les datasets utilisateurs.
    Si item_col est None, sera défini plus tard via unify_book_ids.
    
    Args:
        ratings (pd.DataFrame) : DataFrame avec les utilisateurs, livres et leurs notes.
        user_col (str) : colonne utilisateur à renommer.
        item_col (str) : colonne livre à renommer.
        rating_col (str) : colonne note à renommer.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes renommées.
    """
    ratings = ratings.copy()
    ratings.rename(columns={user_col: "user_id", rating_col: "rating"}, inplace=True)
    if item_col is not None:
        ratings.rename(columns={item_col: "item_id"}, inplace=True)
    return ratings

def unify_book_ids(df):
    """
    Définit la colonne 'item_id' à partir de 'book_id' ou 'ISBN' si item_id n'existe pas.
    """
    df = df.copy()
    if 'item_id' not in df.columns:
        if 'book_id' in df.columns:
            df['item_id'] = df['book_id']
        elif 'ISBN' in df.columns:
            df['item_id'] = df['ISBN']
        else:
            raise ValueError("Le dataset n'a ni 'book_id' ni 'ISBN'.")
    return df

def rename_books_columns(books, book_id_col=None, isbn_col=None, 
                         language_col=None, title_col=None, 
                         author_col=None, publisher_col=None, 
                         year_col=None, image_col=None):
    """
    Renomme les colonnes d'un dataset de livres pour uniformiser.
    Les colonnes peuvent être None si elles n'existent pas.
    Retourne un dataframe avec les noms uniformisés.

    Args:
        books (pd.DataFrame) : le dataFrame dont les colonnes sont à modifier
        book_id_col (str) : colonne qui servira en tant qu'item_id
        isbn_col (str) : colonne isbn qui pourrait servir en tant qu'item_id si book_id_col est None
        title_col (str) : colonne qui servira en tant que title
        author_col (str) : colonne qui servira en tant qu'authors
        publisher_col (str) : colonne qui servira en tant que publisher
        year_col (str) : colonne qui servira en tant que year
        image_col (str) : colonne qui servira en tant qu'image_url

    Returns:
        books (pd.DataFrame) : DataFrame avec les colonnes renommées.
    """
    books = books.copy()
    
    # Définir item_id à partir de book_id ou ISBN
    if book_id_col is not None and book_id_col in books.columns:
        books['item_id'] = books[book_id_col]
    elif isbn_col is not None and isbn_col in books.columns:
        books['item_id'] = books[isbn_col]

    # Mapping des autres colonnes
    rename_map = {}
    if isbn_col is not None: rename_map[isbn_col] = "isbn"
    if title_col is not None: rename_map[title_col] = "title"
    if author_col is not None: rename_map[author_col] = "authors"
    if language_col is not None: rename_map[language_col] = "language"
    if publisher_col is not None: rename_map[publisher_col] = "publisher"
    if year_col is not None: rename_map[year_col] = "year"
    if image_col is not None: rename_map[image_col] = "image_url"

    books.rename(columns=rename_map, inplace=True)
    
    return books

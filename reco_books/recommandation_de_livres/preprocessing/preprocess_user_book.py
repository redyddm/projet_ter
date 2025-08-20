def rename_columns_books(books, ratings, users):
    """
    Sélectionne les colonnes importantes et renomme isbn10 en isbn.
    """
    books.rename(columns={'ISBN':'isbn','Book-Title':'title', 'Book-Author': 'authors','Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)
    books_df = books.drop(columns=['Image-URL-S', 'Image-URL-M'])

    return books_df, ratings, users

def rename_columns_ratings(ratings):

    ratings.rename(columns={'User-ID':'user_id', 'ISBN':'isbn', 'Book-Rating':'rating'}, inplace=True)

    return ratings

def rename_columns_users(users):

    users.rename(columns={'User-ID':'user_id', 'Location':'location','Age':'age'}, inplace=True)

    return users

def fusion_ratings_books(books, ratings):

    ratings_books = ratings.merge(books, on='isbn')

    return ratings_books

def fusion_user_ratings(users, ratings_books):

    users_books=users.merge(ratings_books, on='user_id')
    users_final = users_books[['user_id', 'location', 'age', 'isbn', 'title', 'authors', 'publisher', 'year', 'rating', 'Image-URL-L']].copy()
    
    return users_final
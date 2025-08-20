import difflib
import random
import pandas as pd
import numpy as np

def get_unrated_item_first_version(user_id, ratings):
    unique_isbn = set(ratings['book_id'])

    rated_isbn = set(ratings.loc[ratings['user_id']==user_id, 'book_id'])
    rated_title = set(ratings.loc[ratings['user_id']==user_id, 'title'])

    unrated_isbn = unique_isbn.difference(rated_isbn)

    return unrated_isbn

def get_unrated_item(user_id, ratings):
    
    # dictionnaire isbn -> title
    isbn_to_title = dict(zip(ratings['book_id'], ratings['title']))

    # isbn déjà notés par l'utilisateur
    rated_isbn = set(ratings.loc[ratings['user_id'] == user_id, 'book_id'])
    rated_titles = set(isbn_to_title[i] for i in rated_isbn)

    # tous les isbn
    all_isbn = set(ratings['book_id'])

    # filtrage : garder uniquement les isbn dont le titre n'a pas encore été noté
    unrated_isbn = [isbn for isbn in all_isbn if isbn_to_title[isbn] not in rated_titles]

    return unrated_isbn

def get_index(title, books):
    existing_titles = list(books['title'])
    closest_titles = difflib.get_close_matches(title, existing_titles)
    book_id = books[books['title'] == closest_titles[0]].index[0]
    
    return book_id
    
def get_book(isbn, books):
    match = books[books['book_id'] == isbn]
    if not match.empty:
        return match.iloc[0]['title']  # retourne une chaîne
    return 'Titre inconnu'

def get_book_cover(isbn, books):
    match = books[books['book_id'] == isbn]
    if not match.empty:
        return match.iloc[0]['image_url']  # retourne une chaîne
    return 'Couverture non trouvée'


def get_books_list(isbn_list, books):

    mask=books['book_id'].isin(isbn_list)

    return books[mask].drop_duplicates(subset='book_id').reset_index(drop=True)

def predict_rating(user_id, title, model, data):
    index=get_index(title, data)
    rating=model.predict(uid=user_id, iid=index)

    return rating.est

def predict_rating_book_id(user_id, book_id, model):
    rating=model.predict(uid=user_id, iid=book_id)

    return rating.est

# Recommandation aléatoire

def recommandation_collaborative(user_id, model, ratings, books, note=4):
    non_note=list(get_unrated_item(user_id, ratings))
    random.shuffle(non_note)
    
    for book_id in non_note:
        rating = predict_rating(user_id, book_id, model)
        if rating > note:
            return get_book(book_id, books)
        
def predict_unrated_books(user_id, model, non_note):

    pred_dict = {
        'user_id': user_id,
        'book_id': [],
        'note_predite': []
    }

    for id in non_note:
        pred = model.predict(uid = pred_dict['user_id'],
                                    iid = id)
        pred_dict['book_id'].append(id)
        pred_dict['note_predite'].append(pred.est)

    pred_data = pd.DataFrame(pred_dict).sort_values('note_predite',
                                                     ascending = False)
    
    pred_data = pred_data.drop(columns='user_id')

    return pred_data

def recommandation_collaborative_top_k_gdr(k, user_id, model, ratings):

    non_note=get_unrated_item(user_id, ratings)

    pred_ratings=predict_unrated_books(user_id, model, non_note)

    top_k = pred_ratings.head(k).copy()

    top_k['title'] = top_k['book_id'].apply(get_book, books=ratings)

    top_k['cover'] = top_k['book_id'].apply(get_book_cover, books=ratings)

    return top_k
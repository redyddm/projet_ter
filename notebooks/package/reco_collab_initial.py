import difflib
import random
import pandas as pd
import numpy as np

def get_unrated_item(user_id, ratings):
    unique_isbn = set(ratings['ISBN'])

    rated_isbn = set(ratings.loc[ratings['user_id']==user_id, 'ISBN'])

    unrated_isbn = unique_isbn.difference(rated_isbn)

    return unrated_isbn

def get_index(title, books):
    existing_titles = list(books['title'])
    closest_titles = difflib.get_close_matches(title, existing_titles)
    book_id = books[books['title'] == closest_titles[0]].index[0]
    
    return book_id
    
def get_book(isbn, books):
    match = books[books['ISBN'] == isbn]
    if not match.empty:
        return match.iloc[0]['title']  # retourne une chaîne
    return 'Titre inconnu'


def get_books_list(isbn_list, books):

    mask=books['ISBN'].isin(isbn_list)

    return books[mask].drop_duplicates(subset='ISBN').reset_index(drop=True)

def predict_rating(user_id, title, model, data):
    index=get_index(title, data)
    rating=model.predict(uid=user_id, iid=index)

    return rating.est

def predict_rating_isbn(user_id, isbn, model):
    rating=model.predict(uid=user_id, iid=isbn)

    return rating.est

# Recommandation aléatoire

def recommandation_collaborative(user_id, model, ratings, books, note=4):
    non_note=list(get_unrated_item(user_id, ratings))
    random.shuffle(non_note)
    
    for isbn in non_note:
        rating = predict_rating_isbn(user_id, isbn, model)
        if rating > note:
            return get_book(isbn, books)
        
def predict_unrated_books(user_id, model, non_note):

    pred_dict = {
        'user_id': user_id,
        'ISBN': [],
        'note_predite': []
    }

    for id in non_note:
        pred = model.predict(uid = pred_dict['user_id'],
                                    iid = id)
        pred_dict['ISBN'].append(id)
        pred_dict['note_predite'].append(pred.est)

    pred_data = pd.DataFrame(pred_dict).sort_values('note_predite',
                                                     ascending = False)

    return pred_data

def recommandation_collaborative_top_k(k, user_id, model, ratings, books, ids):
    if user_id not in ids:
        print('Utilisateur pas encore enregistré')
        return 

    non_note=get_unrated_item(user_id, ratings)

    pred_ratings=predict_unrated_books(user_id, model, non_note)

    top_k = pred_ratings.head(k).copy()

    top_k['title'] = top_k['ISBN'].apply(get_book, books=books)

    return top_k
    
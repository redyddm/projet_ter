import difflib
import random
import pandas as pd
import numpy as np

def rescale_ratings(ratings, new_min=1, new_max=5):
    old_min = ratings.min()
    old_max = ratings.max()
    return ((ratings - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

def get_unrated_item_first_version(user_id, ratings):
    unique_item_id = set(ratings['item_id'])

    rated_item_id = set(ratings.loc[ratings['user_id']==user_id, 'item_id'])
    rated_title = set(ratings.loc[ratings['user_id']==user_id, 'title'])

    unrated_item_id = unique_item_id.difference(rated_item_id)

    return unrated_item_id

def get_unrated_item(user_id, ratings):
    
    # dictionnaire item_id -> title
    item_id_to_title = dict(zip(ratings['item_id'], ratings['title']))

    # item_id déjà notés par l'utilisateur
    rated_item_id = set(ratings.loc[ratings['user_id'] == user_id, 'item_id'])
    rated_titles = set(item_id_to_title[i] for i in rated_item_id)

    # tous les item_id
    all_item_id = set(ratings['item_id'])

    # filtrage : garder uniquement les item_id dont le titre n'a pas encore été noté
    unrated_item_id = [item_id for item_id in all_item_id if item_id_to_title[item_id] not in rated_titles]

    return unrated_item_id

def get_index(title, books):
    existing_titles = list(books['title'])
    closest_titles = difflib.get_close_matches(title, existing_titles)
    book_id = books[books['title'] == closest_titles[0]].index[0]
    
    return book_id
    
def get_book(item_id, books):
    match = books[books['item_id'] == item_id]
    if not match.empty:
        return match.iloc[0]  # retourne une chaîne
    return 'Titre inconnu'

def get_book_cover(item_id, books):
    match = books[books['item_id'] == item_id]
    if not match.empty:
        return match.iloc[0]['image_url']  # retourne une chaîne
    return 'Couverture non trouvée'


def get_books_list(item_id_list, books):

    mask=books['item_id'].isin(item_id_list)

    return books[mask].drop_duplicates(subset='item_id').reset_index(drop=True)

def predict_rating(user_id, title, model, data):
    index=get_index(title, data)
    rating=model.predict(uid=user_id, iid=index)

    return rating.est

def predict_rating_item_id(user_id, item_id, model):
    rating=model.predict(uid=user_id, iid=item_id)

    return rating.est

# Recommandation aléatoire

def recommandation_collaborative(user_id, model, ratings, books, note=4):
    non_note=list(get_unrated_item(user_id, ratings))
    random.shuffle(non_note)
    
    for item_id in non_note:
        rating = predict_rating_item_id(user_id, item_id, model)
        if rating > note:
            return get_book(item_id, books)
        
def predict_unrated_books(user_id, model, non_note):

    pred_dict = {
        'user_id': user_id,
        'item_id': [],
        'note_predite': []
    }

    for id in non_note:
        pred = model.predict(uid = pred_dict['user_id'],
                                    iid = id)
        pred_dict['item_id'].append(id)
        pred_dict['note_predite'].append(pred.est)

    pred_data = pd.DataFrame(pred_dict).sort_values('note_predite',
                                                     ascending = False)
    
    pred_data = pred_data.drop(columns='user_id')

    return pred_data

def recommandation_collaborative_top_k(k, user_id, model, ratings, books):

    non_note=get_unrated_item(user_id, ratings)

    pred_ratings=predict_unrated_books(user_id, model, non_note)

    top_k = pred_ratings.head(k).copy()

    top_k_df = top_k['item_id'].apply(get_book, books=ratings)

    return top_k_df, top_k
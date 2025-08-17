import numpy as np
import pandas as pd
import random
import difflib

def get_unrated_item(user_id, ratings):
    unique_title = set(ratings['Name'])

    rated_title = set(ratings.loc[ratings['ID']==user_id, 'Name'])

    unrated_title = unique_title.difference(rated_title)

    return unrated_title

def get_index(title, books):
    existing_titles = list(books['Name'])
    closest_titles = difflib.get_close_matches(title, existing_titles)
    book_id = books[books['Name'] == closest_titles[0]].index[0]
    
    return book_id
def get_book(isbn, books):
    id = np.where(books['ISBN']==isbn)[0]
    if len(id) > 1:
        id=id[0]
    return books.iloc[id]
def get_books_list(isbn_list, books):

    mask=books['ISBN'].isin(isbn_list)

    return books[mask].drop_duplicates(subset='ISBN').reset_index(drop=True)
def predict_rating(user_id, title, model):
    rating=model.predict(uid=user_id, iid=title)

    return rating.est


# Recommandation aléatoire

def recommandation_collaborative(user_id, model, ratings, note=3, n=5):
    non_note=list(get_unrated_item(user_id, ratings))
    random.shuffle(non_note)
    recommandations=[]
    
    for title in non_note:
        rating = predict_rating(user_id, title, model)
        print(rating)
        if rating > note:
            recommandations.append((title, rating))
        if len(recommandations) >= n:
            break
    
    return recommandations

def predict_unrated_books(user_id, model, non_note):

    pred_dict = {
        'ID': user_id,
        'Name': [],
        'note_predite': []
    }

    for id in non_note:
        pred = model.predict(uid = user_id, iid = id)
        #print(pred)
        pred_dict['Name'].append(id)
        pred_dict['note_predite'].append(pred.est)
    
    pred_data = pd.DataFrame(pred_dict).sort_values('note_predite', ascending = False)

    return pred_data

def recommandation_collaborative_top_k(k, user_id, model, ratings, ids):
    if user_id not in ids:
        return 
    non_note=get_unrated_item(user_id, ratings)

    pred_ratings=predict_unrated_books(user_id, model, non_note)

    top_k = pred_ratings.head(k).copy()
    
    return top_k
    
def recommandation_collaborative_top_k_test(k, user_id, model, ratings_train, ratings_test):
    # Récupère les livres que l'utilisateur a notés dans le set de test
    test_items = ratings_test.loc[ratings_test['ID'] == user_id, 'Name'].tolist()
    if len(test_items) == 0:
        return pd.DataFrame(columns=['ID', 'Name', 'note_predite'])  # Pas d'éval possible

    # Prédit uniquement sur ces livres
    pred_dict = {
        'ID': [user_id] * len(test_items),
        'Name': [],
        'note_predite': []
    }
    
    for item in test_items:
        pred = model.predict(uid=user_id, iid=item)
        pred_dict['Name'].append(item)
        pred_dict['note_predite'].append(pred.est)

    pred_data = pd.DataFrame(pred_dict).sort_values('note_predite', ascending=False)
    return pred_data.head(k)

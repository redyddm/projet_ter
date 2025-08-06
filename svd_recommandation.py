import pandas as pd

def get_unrated_item(user_id, ratings):
    unique_title = set(ratings['Name'])

    rated_title = set(ratings.loc[ratings['ID']==user_id, 'Name'])

    unrated_title = unique_title.difference(rated_title)

    return unrated_title

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
    
    pred_data = pd.DataFrame(pred_dict).sort_values('note_predite',
                                                     ascending = False)

    return pred_data
def recommandation_collaborative_top_k(k, user_id, model, ratings, books):
    non_note=get_unrated_item(user_id, ratings)

    pred_ratings=predict_unrated_books(user_id, model, non_note)

    top_k = pred_ratings.head(k).copy()
    
    return top_k
    
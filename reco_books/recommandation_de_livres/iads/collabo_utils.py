import difflib
import random
import pandas as pd
import numpy as np


def rescale_ratings(ratings, new_min=1, new_max=5):
    """
    Remet à l’échelle les notes d’un utilisateur ou d’un dataset dans un nouvel intervalle.

    Args:
        ratings (pd.Series ou np.array) : Notes originales
        new_min (float, optional) : Nouvelle valeur minimale. Defaults to 1.
        new_max (float, optional) : Nouvelle valeur maximale. Defaults to 5.

    Returns:
        pd.Series ou np.array : Notes mises à l’échelle dans [new_min, new_max]
    """
    old_min = ratings.min()
    old_max = ratings.max()
    return ((ratings - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def get_unrated_item(user_id, ratings):
    """
    Retourne les `item_id` que l'utilisateur n'a pas encore notés.

    Args:
        user_id (int ou str) : ID de l'utilisateur
        ratings (pd.DataFrame) : DataFrame contenant au moins ['user_id', 'item_id', 'title']

    Returns:
        list : Liste des item_id non notés par l'utilisateur
    """
    item_id_to_title = dict(zip(ratings['item_id'], ratings['title']))
    rated_item_id = set(ratings.loc[ratings['user_id'] == user_id, 'item_id'])
    rated_titles = set(item_id_to_title[i] for i in rated_item_id)
    all_item_id = set(ratings['item_id'])
    unrated_item_id = [item_id for item_id in all_item_id if item_id_to_title[item_id] not in rated_titles]
    return unrated_item_id


def get_index(title, books):
    """
    Retourne l’index d’un livre dans le DataFrame `books` en cherchant le titre le plus proche.

    Args:
        title (str) : Titre à rechercher
        books (pd.DataFrame) : DataFrame contenant la colonne 'title'

    Returns:
        int : Index du livre correspondant
    """
    existing_titles = list(books['title'])
    closest_titles = difflib.get_close_matches(title, existing_titles)
    book_id = books[books['title'] == closest_titles[0]].index[0]
    return book_id


def get_book(item_id, books):
    """
    Retourne la ligne correspondant à un livre donné à partir de son item_id.

    Args:
        item_id (int ou str) : ID du livre
        books (pd.DataFrame) : DataFrame contenant au moins ['item_id']

    Returns:
        pd.Series ou str : Ligne du DataFrame correspondant au livre ou 'Titre inconnu' si non trouvé
    """
    match = books[books['item_id'] == item_id]
    if not match.empty:
        return match.iloc[0]
    return 'Titre inconnu'


def predict_unrated_books(user_id, model, non_note):
    """
    Prédit les notes pour les livres non encore notés par un utilisateur.

    Args:
        user_id (int ou str) : ID de l'utilisateur
        model (Surprise model) : Modèle entraîné pouvant prédire avec `.predict(uid, iid)`
        non_note (list) : Liste d’item_id non notés par l’utilisateur

    Returns:
        pd.DataFrame : DataFrame triée par note prédite décroissante avec colonnes ['item_id', 'note_predite']
    """
    pred_dict = {'user_id': user_id, 'item_id': [], 'note_predite': []}
    for id in non_note:
        pred = model.predict(uid=user_id, iid=id)
        pred_dict['item_id'].append(id)
        pred_dict['note_predite'].append(pred.est)
    pred_data = pd.DataFrame(pred_dict).sort_values('note_predite', ascending=False)
    pred_data = pred_data.drop(columns='user_id')
    return pred_data


def recommandation_collaborative_top_k(k, user_id, model, ratings, books):
    """
    Retourne les top-K recommandations pour un utilisateur avec un modèle collaboratif.

    Args:
        k (int) : Nombre de recommandations à retourner
        user_id (int ou str) : ID de l'utilisateur
        model (Surprise model) : Modèle entraîné
        ratings (pd.DataFrame) : DataFrame des notes avec éventuellement la colonne 'title'
        books (pd.DataFrame) : DataFrame des livres avec 'item_id' et autres métadonnées

    Returns:
        tuple:
            - top_k_df (pd.Series) : Titre ou ligne des livres recommandés
            - top_k (pd.DataFrame) : DataFrame contenant ['item_id', 'note_predite'] pour les top K
    """
    non_note = get_unrated_item(user_id, ratings)
    pred_ratings = predict_unrated_books(user_id, model, non_note)
    top_k = pred_ratings.head(k).copy()
    if 'title' in ratings.columns:
        top_k_df = top_k['item_id'].apply(get_book, books=ratings)
    else:
        top_k_df = top_k['item_id'].apply(get_book, books=books)
    return top_k_df, top_k

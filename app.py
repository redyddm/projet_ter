import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_books():
    return pd.read_pickle('./datasets/reco_datasets/content_dataset_desc_final.pkl')

@st.cache_data
def load_ratings():
    return pd.read_pickle('./datasets/reco_datasets/ratings_books_final.pkl')

st.title('Recommandation de livres')

with st.spinner('Chargement des données...'):
    books = load_books()
    ratings = load_ratings()

st.success('Données chargées !')

# Liste des utilisateurs uniques
user_ids = ratings['user_id'].unique()
selected_user = st.selectbox('Sélectionnez un utilisateur', user_ids)

# Filtrer les notes de cet utilisateur
user_ratings = ratings[ratings['user_id'] == selected_user]

st.subheader(f"Notes de l'utilisateur {selected_user}")

if not user_ratings.empty:
    # On peut joindre avec le dataset des livres pour afficher les titres
    user_ratings
else:
    st.write("Cet utilisateur n'a aucune note enregistrée.")

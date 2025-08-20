from pathlib import Path
import pandas as pd
import pickle
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_csv, load_pkl
from recommandation_de_livres.iads.svd_utils_gdr import (
    recommandation_collaborative_top_k_gdr,
)

DIR = 'goodreads'

# Titre de l'application
st.title("Recommandation de livres - Collaborative (SVD)")

# Chemins
ratings_path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.csv"
model_path = MODELS_DIR / DIR / "svd_model.pkl"

# Cache chargement dataset
@st.cache_data
def load_ratings(path):
    return load_csv(path)

ratings = load_ratings(ratings_path)

# Cache chargement modèle
@st.cache_resource
def load_model(path):
    return load_pkl(path)

model = load_model(model_path)

# Choix utilisateur
user_index = st.selectbox(
    "Sélectionnez un utilisateur :", 
    sorted(ratings["user_index"].unique())
)
user_id = ratings.loc[ratings["user_index"] == user_index, "user_id"].iloc[0]
top_k = st.slider("Nombre de recommandations souhaité", min_value=1, max_value=20, value=5)

# Recommandations
if st.button("Rechercher"):

    top_books = recommandation_collaborative_top_k_gdr(
        k=top_k,
        user_id=user_id,
        model=model,
        ratings=ratings
    )

    images=top_books['cover']
    top_books = top_books.drop(columns='cover')

    st.subheader(f"Top {top_k} recommandations pour l'utilisateur {user_index}")

    # Affichage des recommandations sous forme de tableau
    st.dataframe(top_books)

    cols = st.columns(len(top_books))

    for col, img_url in zip(cols, images):
        if img_url != '[]':
            col.image(img_url, width=150) 



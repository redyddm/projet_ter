from pathlib import Path
import pandas as pd
import pickle
import streamlit as st

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.iads.svd_utils import (
    recommandation_collaborative_top_k,
)

# Titre de l'application
st.title("Recommandation de livres - Collaborative (SVD)")

# Chemins
ratings_path = PROCESSED_DATA_DIR / "collaborative_dataset.csv"
model_path = MODELS_DIR / "svd_model.pkl"

# Cache chargement dataset et modèle
@st.cache_data
def load_ratings(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

ratings = load_ratings(ratings_path)
model = load_model(model_path)

# Choix utilisateur
user_id = st.selectbox(
    "Sélectionnez un utilisateur :", 
    sorted(ratings["user_id"].unique())
)
top_k = st.slider("Nombre de recommandations souhaité", min_value=1, max_value=20, value=5)

# Recommandations
if st.button("Rechercher"):
    if user_id not in ratings["user_id"].unique():
        st.warning(f"⚠️ L'utilisateur {user_id} n'existe pas dans le dataset.")
    else:

        top_books = recommandation_collaborative_top_k(
            k=top_k,
            user_id=user_id,
            model=model,
            ratings=ratings
        )

        images=top_books['cover']
        top_books = top_books.drop(columns='cover')

        st.subheader(f"Top {top_k} recommandations pour l'utilisateur {user_id}")

        # Affichage des recommandations sous forme de tableau
        st.dataframe(top_books)


        cols = st.columns(len(top_books))

        for col, img_url in zip(cols, images):
            if img_url != '[]':
                col.image(img_url, width=150) 



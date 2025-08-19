import streamlit as st
import pandas as pd
import numpy as np
import gensim
import pickle
import requests
from sentence_transformers import SentenceTransformer

from pathlib import Path
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.content_utils import recommandation_content_top_k
from recommandation_de_livres.iads.svd_utils import recommandation_collaborative_top_k

# -------------------------
# Titre
# -------------------------
st.title("Recommandation de livres - Multi-modèle (Galerie + Pagination)")

# -------------------------
# Chargement des données
# -------------------------
@st.cache_data
def load_books():
    return pd.read_pickle(PROCESSED_DATA_DIR / "content_dataset.pkl")

@st.cache_data
def load_ratings():
    return pd.read_csv(PROCESSED_DATA_DIR / "collaborative_dataset.csv")

@st.cache_resource
def load_w2v_model():
    return gensim.models.Word2Vec.load(str(MODELS_DIR / "word2vec.model"))

@st.cache_data
def load_w2v_embeddings():
    return np.load(PROCESSED_DATA_DIR / "embeddings_w2v.npy")

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer(str(MODELS_DIR / "sbert_model/"))

@st.cache_data
def load_sbert_embeddings():
    return np.load(PROCESSED_DATA_DIR / "embeddings_sbert.npy")

@st.cache_resource
def load_svd_model():
    with open(MODELS_DIR / "svd_model.pkl", "rb") as f:
        return pickle.load(f)

books = load_books()
ratings = load_ratings()
w2v_model = load_w2v_model()
w2v_embeddings = load_w2v_embeddings()
sbert_model = load_sbert_model()
sbert_embeddings = load_sbert_embeddings()
svd_model = load_svd_model()

# -------------------------
# Choix du modèle et top-k
# -------------------------
model_choice = st.radio("Choisissez le modèle", ["SVD", "Word2Vec", "SBERT"])
top_k = st.number_input("Nombre de recommandations", min_value=1, max_value=50, value=10)

# -------------------------
# Inputs utilisateur
# -------------------------
if model_choice == "Word2Vec":
    title_input = st.text_input("Entrez le titre du livre", "Harry Potter")
elif model_choice == "SBERT":
    title_input = st.text_input("Entrez un mot-clé ou description", "magie")
elif model_choice == "SVD":
    user_id = st.selectbox("Sélectionnez un utilisateur", sorted(ratings["user_id"].unique()))

# -------------------------
# Recommandations
# -------------------------
if st.button("Rechercher"):

    if model_choice == "Word2Vec":
        top_books = recommandation_content_top_k(title_input, w2v_embeddings, w2v_model, books, top_k)
        images = []
        for _, row in top_books.iterrows():
            isbn = row.get('isbn13')
            cover_url = f"https://bookcover.longitood.com/bookcover/{isbn}"
            try:
                resp = requests.get(cover_url)
                images.append(resp.json().get("url") if resp.status_code==200 else None)
            except:
                images.append(None)

    elif model_choice == "SBERT":
        top_books = recommandation_content_top_k(title_input, sbert_embeddings, sbert_model, books, top_k)
        images = []
        for _, row in top_books.iterrows():
            isbn = row.get('isbn13')
            cover_url = f"https://bookcover.longitood.com/bookcover/{isbn}"
            try:
                resp = requests.get(cover_url)
                images.append(resp.json().get("url") if resp.status_code==200 else None)
            except:
                images.append(None)

    elif model_choice == "SVD":
        top_books = recommandation_collaborative_top_k(k=top_k, user_id=user_id, model=svd_model, ratings=ratings)
        images = top_books['cover'].tolist()
        top_books = top_books.drop(columns='cover')

    st.subheader(f"Top {top_k} recommandations - Modèle : {model_choice}")

    # -------------------------
    # Pagination interactive
    # -------------------------
    books_per_page = 5
    total_pages = (len(top_books) + books_per_page - 1) // books_per_page

    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("Précédent"):
            if st.session_state.page_number > 0:
                st.session_state.page_number -= 1
    with col3:
        if st.button("Suivant"):
            if st.session_state.page_number < total_pages - 1:
                st.session_state.page_number += 1

    start_idx = st.session_state.page_number * books_per_page
    end_idx = min(start_idx + books_per_page, len(top_books))
    current_books = top_books.iloc[start_idx:end_idx]
    current_images = images[start_idx:end_idx]

    cols = st.columns(len(current_books))
    for col, (_, row), img_url in zip(cols, current_books.iterrows(), current_images):
        with col:
            if img_url:
                st.image(img_url, width=120)
            st.markdown(f"**{row.get('title','')}**")
            st.markdown(f"*{row.get('authors','')}*")

    st.markdown(f"Page {st.session_state.page_number + 1} / {total_pages}")

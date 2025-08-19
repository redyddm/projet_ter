import streamlit as st
import pandas as pd
import numpy as np
import gensim
from sentence_transformers import SentenceTransformer
import requests
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.content_utils import recommandation_content_top_k, suggest_titles
from recommandation_de_livres.iads.svd_utils import recommandation_collaborative_top_k

# -------------------------
# Chargement des données
# -------------------------
@st.cache_data
def load_books():
    return pd.read_pickle(PROCESSED_DATA_DIR / "content_dataset.pkl")

@st.cache_data
def load_ratings():
    return pd.read_csv(PROCESSED_DATA_DIR / "collaborative_dataset.csv")

books = load_books()
ratings = load_ratings()

# -------------------------
# Chargement TF-IDF pré-calculé
# -------------------------
@st.cache_resource
def load_tfidf():
    with open(MODELS_DIR / "tfidf_model.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = load_tfidf()

nb_sugg = 10

# -------------------------
# Choix du modèle
# -------------------------
model_choice = st.radio("Choisissez le modèle", ["Word2Vec", "SBERT", "SVD"])
top_k = st.number_input("Nombre de recommandations", min_value=1, max_value=20, value=5)

# -------------------------
# Lazy loading pour Word2Vec / SBERT / SVD
# -------------------------
@st.cache_resource
def load_w2v_model_and_embeddings():
    w2v_model = gensim.models.Word2Vec.load(str(MODELS_DIR / "word2vec.model"))
    w2v_embeddings = np.load(PROCESSED_DATA_DIR / "embeddings_w2v.npy", mmap_mode='r')
    return w2v_model, w2v_embeddings

@st.cache_resource
def load_sbert_model_and_embeddings():
    sbert_model = SentenceTransformer(str(MODELS_DIR / "sbert_model/"))
    sbert_embeddings = np.load(PROCESSED_DATA_DIR / "embeddings_sbert.npy", mmap_mode='r')
    return sbert_model, sbert_embeddings

@st.cache_resource
def load_svd_model():
    with open(MODELS_DIR / "svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    return svd_model

if model_choice == "Word2Vec":
    model, embeddings = load_w2v_model_and_embeddings()
elif model_choice == "SBERT":
    model, embeddings = load_sbert_model_and_embeddings()
elif model_choice == "SVD":
    svd_model = load_svd_model()

# -------------------------
# Input utilisateur
# -------------------------
user_input = None
selected_title = None
user_id = None

if model_choice in ["Word2Vec", "SBERT"]:
    user_input = st.text_input("Tapez un titre ou mot-clé")
    if user_input:
        suggestions_df = suggest_titles(user_input, tfidf, tfidf_matrix, books, k=nb_sugg)
        suggestion_list = suggestions_df['title'] + " - " + suggestions_df['authors']
        selected_title_author = st.selectbox("Titres suggérés :", suggestion_list)
        selected_title = selected_title_author.split(" - ")[0]

elif model_choice == "SVD":
    user_id = st.selectbox("Sélectionnez un utilisateur", sorted(ratings['user_id'].unique()))

# -------------------------
# Recommandations
# -------------------------
if st.button("Rechercher"):
    top_books = None

    if model_choice in ["Word2Vec", "SBERT"] and selected_title:
        top_books = recommandation_content_top_k(selected_title, embeddings, model, books, top_k)
        # Préparer les images
        if 'Image-URL-L' in top_books.columns:
            images = top_books['Image-URL-L'].tolist() if 'Image-URL-L' in top_books.columns else [None]*len(top_books)
            top_books = top_books.drop(columns='Image-URL-L', errors='ignore')
        else:
            images = []
            for _, row in top_books.iterrows():
                isbn = row.get('isbn13') or row.get('isbn')
                cover_url = f"https://bookcover.longitood.com/bookcover/{isbn}"
                try:
                    response = requests.get(cover_url)
                    if response.status_code == 200:
                        data = response.json()
                        images.append(data.get("url"))
                    else:
                        images.append(None)
                except:
                    images.append(None)

    elif model_choice == "SVD" and user_id is not None:
        top_books = recommandation_collaborative_top_k(top_k, user_id, svd_model, ratings)
        images = top_books['cover'].tolist() if 'cover' in top_books.columns else [None]*len(top_books)
        top_books = top_books.drop(columns='cover', errors='ignore')

    if top_books is not None:
        if images:
            cols = st.columns(len(top_books))
            for col, img_url, (_, row) in zip(cols, images, top_books.iterrows()):
                if img_url:
                    col.image(img_url, width=150)
                col.markdown(f"**{row.get('title','')}**  \n*{row.get('authors','')}*")
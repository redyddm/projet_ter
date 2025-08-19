from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import requests

from sentence_transformers import SentenceTransformer

from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.content_utils import recommandation_content_top_k

# Titre de l'application
st.title("Recommandation de livres - Contenu (Sentence-BERT)")

# Inputs utilisateur
title = st.text_input("Entrez le titre du livre", "Harry Potter")
top_k = st.slider("Nombre de recommandations souhaité", min_value=1, max_value=20, value=5)

# Chargement des fichiers
content_path = PROCESSED_DATA_DIR / "content_dataset.pkl"
model_sbert_path = MODELS_DIR / "sbert_model/"
embeddings_sbert_path = PROCESSED_DATA_DIR / "embeddings_sbert.npy"

@st.cache_data
def load_content(path):
    return pd.read_pickle(path)

@st.cache_resource
def load_model(model_path):
    return SentenceTransformer(str(model_path))

@st.cache_data
def load_embeddings(path):
    return np.load(path)

content_df = load_content(content_path)
model = load_model(model_sbert_path)
embeddings = load_embeddings(embeddings_sbert_path)

# Recommandations
if st.button("Rechercher"):
    
    top_books = recommandation_content_top_k(title, embeddings, model, content_df, top_k)

    st.subheader(f"Top {top_k} recommandations pour : {title}")

    images = []
    for _, row in top_books.iterrows():
        isbn = row['isbn13']
        cover_url = f"https://bookcover.longitood.com/bookcover/{isbn}"

        # Vérifier si l'image existe
        response = requests.get(cover_url)
        if response.status_code == 200:
            data = response.json() 
            images.append(data.get("url"))
        else:
            images.append("[]")
        st.markdown(f"**{row['title']}** - *{row['authors']}*")
    
    cols = st.columns(len(images))

    for col, img_url in zip(cols, images):
        if img_url != '[]':
            col.image(img_url, width=150)  # largeur fixe pour harmoniser

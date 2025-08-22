import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from pathlib import Path
from loguru import logger
import pandas as pd
from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR

st.title("🧠 Génération d'embeddings SBERT")

# --- Choix du dataset ---
dataset_choice = st.radio(
    "Choisir le dataset :", 
    ["Recommender", "Depository", "Goodreads"], 
    index=2
)
DIR = dataset_choice.lower()

# --- Chargement des données ---
features_path = PROCESSED_DATA_DIR / DIR / "features_sbert.parquet"
content_df = load_parquet(features_path)
st.success(f"Dataset chargé ({len(content_df)} livres)")

# --- Choix du modèle SBERT ---
st.subheader("⚙️ Choix du modèle SBERT")
model_options = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
    "distiluse-base-multilingual-cased-v2",
    "paraphrase-multilingual-MiniLM-L12-v2"
]

selected_model = st.selectbox("Modèles recommandés :", model_options, index=0)
custom_model = st.text_input("Ou entrez un modèle HuggingFace personnalisé :", "")

model_name = custom_model.strip() if custom_model.strip() != "" else selected_model
st.write(f"✅ Modèle choisi : {model_name}")

# --- Paramètres ---
batch_size = st.number_input("Batch size", min_value=16, max_value=256, value=64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"💻 Device utilisé : {device}")

# --- Génération d'embeddings ---
if st.button("Créer les embeddings SBERT"):
    st.info("🔄 Chargement du modèle...")
    sbert = SentenceTransformer(model_name, device=device)
    
    st.info("🔄 Génération des embeddings...")
    embeddings = sbert.encode(
        content_df['text_clean'], 
        convert_to_numpy=True, 
        batch_size=batch_size, 
        show_progress_bar=True
    )
    
    # Sauvegarde
    model_path = MODELS_DIR / DIR / "sbert_model"
    embeddings_path = PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy"

    st.info("💾 Sauvegarde du modèle et des embeddings...")
    sbert.save(str(model_path))
    np.save(embeddings_path, embeddings)

    st.success("✅ Embeddings SBERT générés et sauvegardés !")
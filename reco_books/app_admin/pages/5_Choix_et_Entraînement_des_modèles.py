import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from surprise import SVD, Dataset, Reader
from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_parquet

st.title("⚙️ Entraînement du modèle SVD")

# --- Choix du dataset ---
DIR = st.session_state.get("DIR", None)

if not DIR:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à l'accueil.")
    st.stop()

features_path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet"
if not features_path.exists():
    st.error(f"Fichier introuvable : {features_path}")
    st.stop()

st.title(f"🧠 Génération des Features - {DIR}")

model_path = MODELS_DIR / DIR / "collaborative_model.pkl"

# --- Hyperparamètres ---
st.subheader("🔧 Hyperparamètres du modèle")
n_factors = st.number_input("Nombre de facteurs latents", min_value=10, max_value=500, value=50, step=10)
n_epochs = st.number_input("Nombre d'epochs", min_value=5, max_value=200, value=50, step=5)
lr_all = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.002, step=0.001, format="%.4f")
reg_all = st.number_input("Régularisation", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.4f")

# --- Choix du rating scale ---
st.subheader("📏 Échelle des notes")
min_rating = st.number_input("Note minimale", min_value=1.0, max_value=10.0, value=1.0, step=0.5)
max_rating = st.number_input("Note maximale", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

if min_rating >= max_rating:
    st.error("⚠️ La note minimale doit être inférieure à la note maximale.")
    st.stop()

rating_scale = (min_rating, max_rating)

# --- Chargement des données ---
st.info("Chargement des données...")
collaborative_df = load_parquet(features_path)
st.write(f"📊 {len(collaborative_df)} notes chargées.")

# --- Bouton entraînement ---
if st.button("🚀 Lancer l'entraînement"):
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(collaborative_df[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    svd = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all
    )

    with st.spinner("Entraînement en cours... ⏳"):
        svd.fit(trainset)

    # Sauvegarde du modèle
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(svd, f)

    st.success(f"✅ Modèle entraîné et sauvegardé dans {model_path}")
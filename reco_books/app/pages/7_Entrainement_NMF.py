import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from surprise import NMF, Dataset, Reader
from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_parquet

st.title("⚙️ Entraînement du modèle NMF")

# --- Choix du dataset ---
choice = st.radio("Choix du dataset :", ["Recommender (1)", "Goodreads (2)"], index=1)

if choice.startswith("Recommender"):
    DIR = "recommender"
    rating_scale = (1, 10)
    id_col = "ISBN"
else:
    DIR = "goodreads"
    rating_scale = (1, 5)
    id_col = "book_id"

features_path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet"
model_path = MODELS_DIR / DIR / "nmf_model.pkl"

# --- Hyperparamètres ---
st.subheader("🔧 Hyperparamètres du modèle")
n_factors = st.number_input("Nombre de facteurs latents", min_value=5, max_value=500, value=50, step=5)
n_epochs = st.number_input("Nombre d'epochs", min_value=5, max_value=200, value=50, step=5)
reg_pu = st.number_input("Régularisation utilisateurs (reg_pu)", min_value=0.0001, max_value=1.0, value=0.06, step=0.01, format="%.4f")
reg_qi = st.number_input("Régularisation items (reg_qi)", min_value=0.0001, max_value=1.0, value=0.06, step=0.01, format="%.4f")

# --- Chargement des données ---
st.info("Chargement des données...")
collaborative_df = load_parquet(features_path)
st.write(f"📊 {len(collaborative_df)} notes chargées.")

# --- Bouton entraînement ---
if st.button("🚀 Lancer l'entraînement"):
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(collaborative_df[['user_id', id_col, 'rating']], reader)
    trainset = data.build_full_trainset()

    nmf = NMF(
        n_factors=n_factors,
        n_epochs=n_epochs,
        reg_pu=reg_pu,
        reg_qi=reg_qi
    )

    with st.spinner("Entraînement en cours... ⏳"):
        nmf.fit(trainset)

    # Sauvegarde du modèle
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(nmf, f)

    st.success(f"✅ Modèle entraîné et sauvegardé dans {model_path}")
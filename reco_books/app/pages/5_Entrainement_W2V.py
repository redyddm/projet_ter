import streamlit as st
import numpy as np
import gensim
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.iads.content_utils import get_text_vector
from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.iads.progress_w2v import TqdmCorpus, EpochLogger

st.title("📖 Entraînement Word2Vec")

# --- Choix du dataset ---
choice = st.radio(
    "Choix du dataset :",
    ["Recommender (1)", "Depository (2)", "Goodreads (3)"],
    index=2
)

DIR = {"Recommender (1)": "recommender",
       "Depository (2)": "depository",
       "Goodreads (3)": "goodreads"}[choice]

features_path = PROCESSED_DATA_DIR / DIR / "features_w2v.parquet"
model_path = MODELS_DIR / DIR / "word2vec.model"
embeddings_path = PROCESSED_DATA_DIR / DIR / "embeddings_w2v.npy"

# --- Hyperparamètres ---
st.subheader("🔧 Hyperparamètres")
vector_size = st.number_input("Dimension des vecteurs (vector_size)", min_value=50, max_value=1000, value=300, step=50)
window = st.number_input("Fenêtre de contexte (window)", min_value=2, max_value=20, value=10, step=1)
min_count = st.number_input("Fréquence minimale des mots (min_count)", min_value=1, max_value=10, value=2, step=1)
epochs = st.number_input("Nombre d'epochs", min_value=1, max_value=100, value=5, step=1)
sg = st.number_input("CBOW (0, par défaut) , Skip-Gram (1)", min_value=0, max_value=1, value=0, step=1)

# --- Chargement des données et corpus ---
if "content_df" not in st.session_state:
    st.info("Chargement des textes nettoyés...")
    st.session_state.content_df = load_parquet(features_path)
    st.session_state.corpus = TqdmCorpus(
        st.session_state.content_df['text_clean'].apply(lambda x: list(x))
    )
    st.success(f"📊 {len(st.session_state.content_df)} documents chargés et corpus créé.")

# --- Bouton entraînement ---
if st.button("🚀 Lancer l'entraînement Word2Vec"):
    st.write("Création du modèle Word2Vec...")
    w2v = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=mp.cpu_count(),
        sg=sg
    )

    st.write("Construction du vocabulaire...")
    w2v.build_vocab(st.session_state.corpus)

    st.write("🔄 Entraînement en cours...")
    with st.spinner("Entraînement Word2Vec..."):
        w2v.train(
            st.session_state.corpus,
            total_examples=w2v.corpus_count,
            epochs=epochs,
            callbacks=[EpochLogger()]
        )

    st.success("✅ Entraînement terminé !")

    # --- Sauvegarde du modèle ---
    model_path.parent.mkdir(parents=True, exist_ok=True)
    w2v.save(str(model_path))
    st.success(f"📂 Modèle sauvegardé dans {model_path}")

    # --- Calcul embeddings ---
    st.write("Calcul des embeddings...")
    book_embeddings = np.vstack([
        get_text_vector(tokens, w2v) for tokens in tqdm(st.session_state.corpus, desc="Calcul embeddings")
    ])
    np.save(embeddings_path, book_embeddings)
    st.success(f"📂 Embeddings sauvegardés dans {embeddings_path}")

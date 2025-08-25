import streamlit as st
import pandas as pd
import numpy as np
import gensim
import torch
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.iads.content_utils import combine_text, get_text_vector
from recommandation_de_livres.iads.text_cleaning import nettoyage_texte
from recommandation_de_livres.iads.progress_w2v import TqdmCorpus, EpochLogger
import multiprocessing as mp

# --- Sélection du dataset ---
if 'DIR' not in st.session_state:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à la page d'accueil pour en choisir un.")
    st.stop()

# -------------------------------
# Dataset déjà choisi à l'accueil
# -------------------------------
DIR = st.session_state['DIR']
dataset_path = PROCESSED_DATA_DIR / DIR

st.title(f"🛠️ Entrainement des modèles basés sur le contenu - {DIR}")

tabs = st.tabs(["Word2Vec", "Sentence-BERT", "TF-IDF"])

# ------------------ WORD2VEC ------------------
with tabs[0]:
    st.subheader("🔤 Features Word2Vec")
    st.info("Nettoie, tokenise et prépare les textes pour un entraînement Word2Vec.")

    vector_size = st.number_input("Dimension des vecteurs", min_value=50, max_value=1024, value=300, step=50)
    window = st.number_input("Taille de la fenêtre", min_value=2, max_value=20, value=10, step=1)
    min_count = st.number_input("min_count", min_value=1, max_value=10, value=2, step=1)
    epochs = st.number_input("Nombre d'epochs", min_value=1, max_value=50, value=5, step=1)

    if st.button("🚀 Lancer Word2Vec"):
        progress = st.progress(0, text="Initialisation...")
        try:
            content_df = load_parquet(dataset_path / "features_w2v.parquet")
            progress.progress(10, text="Chargement du dataset...")

            corpus = TqdmCorpus(content_df['text_clean'].apply(lambda x: list(x)))
            w2v = gensim.models.Word2Vec(
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=mp.cpu_count(),
                sg=0
            )

            progress.progress(30, text="Construction du vocabulaire...")
            w2v.build_vocab(corpus)

            progress.progress(50, text="Entraînement du modèle...")
            w2v.train(corpus, total_examples=w2v.corpus_count, epochs=epochs, callbacks=[EpochLogger()])

            model_path = MODELS_DIR / DIR / "word2vec.model"
            embeddings_path = PROCESSED_DATA_DIR / DIR / "embeddings_w2v.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)

            w2v.save(str(model_path))
            progress.progress(70, text="Modèle sauvegardé...")

            book_embeddings = np.vstack([get_text_vector(tokens, w2v) for tokens in corpus])
            np.save(embeddings_path, book_embeddings)
            progress.progress(100, text="Embeddings sauvegardés ✅")

            st.success("Word2Vec terminé !")
            st.download_button("Télécharger le modèle Word2Vec", data=open(model_path,'rb').read(), file_name="word2vec.model")
            st.download_button("Télécharger les embeddings", data=open(embeddings_path,'rb').read(), file_name="embeddings_w2v.npy")

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# ------------------ SENTENCE-BERT ------------------
with tabs[1]:
    st.subheader("🤖 Features Sentence-BERT")
    st.info("Génère des embeddings Sentence-BERT pour chaque livre à partir du texte nettoyé.")

    batch_size = st.number_input("Batch size", min_value=8, max_value=256, value=64, step=8)

    if st.button("🚀 Lancer SBERT"):
        progress = st.progress(0, text="Initialisation...")
        try:
            content_df = load_parquet(dataset_path / "features_sbert.parquet")
            progress.progress(10, text="Chargement des features...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            progress.progress(30, text=f"Modèle chargé sur {device}")

            embeddings = sbert.encode(content_df['text_clean'], convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
            progress.progress(80, text="Encodage terminé")

            model_path = MODELS_DIR / DIR / "sbert_model"
            embeddings_path = PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)

            sbert.save(str(model_path))
            np.save(embeddings_path, embeddings)
            progress.progress(100, text="Terminé ✅")

            st.success("SBERT terminé !")
            st.download_button("Télécharger le modèle SBERT", data=open(model_path / "config.json",'rb').read(), file_name="sbert_model.zip")
            st.download_button("Télécharger les embeddings", data=open(embeddings_path,'rb').read(), file_name="embeddings_sbert.npy")

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# ------------------ TF-IDF ------------------
with tabs[2]:
    st.subheader("📝 Features TF-IDF")
    st.info("Génère une matrice TF-IDF sur les textes nettoyés (titre + auteurs).")

    ngram_min = st.number_input("Ngram min", min_value=1, max_value=3, value=1)
    ngram_max = st.number_input("Ngram max", min_value=1, max_value=3, value=3)

    if st.button("🚀 Lancer TF-IDF"):
        progress = st.progress(0, text="Initialisation...")
        try:
            features_df = load_parquet(dataset_path / "features_tfidf.parquet")
            progress.progress(10, text="Chargement du dataset...")

            tfidf = TfidfVectorizer(ngram_range=(ngram_min, ngram_max), lowercase=True, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(features_df['text_clean'])
            progress.progress(70, text="TF-IDF entraîné")

            model_path = MODELS_DIR / DIR / "tfidf_model.pkl"
            matrix_path = PROCESSED_DATA_DIR / DIR / "tfidf_matrix.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            matrix_path.parent.mkdir(parents=True, exist_ok=True)

            with open(model_path,"wb") as f:
                pickle.dump(tfidf,f)
            with open(matrix_path,"wb") as f:
                pickle.dump(tfidf_matrix,f)
            progress.progress(100, text="TF-IDF sauvegardé ✅")

            st.success("TF-IDF terminé !")
            st.download_button("Télécharger le modèle TF-IDF", data=open(model_path,'rb').read(), file_name="tfidf_model.pkl")
            st.download_button("Télécharger la matrice TF-IDF", data=open(matrix_path,'rb').read(), file_name="tfidf_matrix.pkl")

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

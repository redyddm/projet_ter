import streamlit as st
import pandas as pd
import sys
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import gensim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.utils import stars
from recommandation_de_livres.loaders.load_data import load_pkl, load_parquet
from recommandation_de_livres.iads.svd_utils_gdr import recommandation_collaborative_top_k_gdr
from recommandation_de_livres.iads.content_utils import suggest_titles, recommandation_content_top_k_gdr

DIR = "goodreads"

# ---------------------------
# Vérifier connexion
# ---------------------------
if not st.session_state.get("logged_in", False):
    st.warning("🚪 Veuillez vous connecter pour accéder à cette page.")
    st.stop()

books = st.session_state["books"]
ratings = st.session_state["ratings"]
users = st.session_state["users"]

@st.cache_data
def load_content():
    path = PROCESSED_DATA_DIR / DIR / "content_dataset.parquet"
    return load_parquet(path)

@st.cache_resource
def load_tfidf():
    with open(MODELS_DIR / DIR / "tfidf_model.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(PROCESSED_DATA_DIR / DIR / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = load_tfidf()
content_df=load_content()

# ---------------------------
# Choix du type de recommandation
# ---------------------------
reco_type = st.selectbox(
    "Type de recommandation",
    options=["Collaborative (SVD)", "Contenu (Word2Vec)", "Contenu (Sentence-BERT)"]
)

top_k = st.slider("Nombre de recommandations", 1, 20, 5)

# ---------------------------
# Recommandations SVD
# ---------------------------
if reco_type == "Collaborative (SVD)":

    @st.cache_resource
    def load_svd_model(model_name):
        path = MODELS_DIR / model_name / "svd_model.pkl"
        return load_pkl(path)

    model = load_svd_model(DIR)

    if st.button("Rechercher SVD"):
        top_books = recommandation_collaborative_top_k_gdr(
            k=top_k,
            user_id=st.session_state["user_id"],
            model=model,
            ratings=ratings
        )

        cols = st.columns(5)
        for i, (_, book) in enumerate(top_books.iterrows()):
            col = cols[i % 5]
            with col:
                st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
                st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
                st.markdown(stars(book.get("predicted_rating", 0)))
                st.caption(book.get("authors", "Auteur inconnu"))

                with st.expander("📖 Voir détails"):
                    st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                    st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                    st.write(f"**Année :** {book.get('year', 'Inconnue')}")
                    st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                    st.markdown("**Description :**")
                    st.write(book.get("description", "Pas de description disponible."))

# ---------------------------
# Recommandations Word2Vec
# ---------------------------
elif reco_type == "Contenu (Word2Vec)":
    @st.cache_resource
    def load_w2v_model():
        path = MODELS_DIR / DIR / "word2vec.model"
        return gensim.models.Word2Vec.load(str(path))

    @st.cache_data
    def load_embeddings_w2v():
        path = PROCESSED_DATA_DIR / DIR / "embeddings_w2v.npy"
        return np.load(path)
    
    w2v_model = load_w2v_model()
    embeddings_w2v = load_embeddings_w2v()

    nb_sugg = 10

    book_title = st.text_input("Titre du livre pour la recommandation Word2Vec")

    selected_title = None
    if book_title:
        suggestions_df = suggest_titles(book_title, tfidf, tfidf_matrix, content_df, k=nb_sugg)
        suggestion_list = suggestions_df['title'] + " - " + suggestions_df['authors']
        selected_title_author = st.selectbox("Titres suggérés :", suggestion_list)
        selected_title = selected_title_author.split(" - ")[0]

    if st.button("Rechercher Word2Vec"):
        top_books = recommandation_content_top_k_gdr(selected_title, embeddings_w2v, w2v_model, content_df, top_k)

        cols = st.columns(5)
        for i, (_, book) in enumerate(top_books.iterrows()):
            col = cols[i % 5]
            with col:
                st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
                st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
                st.caption(book.get("authors", "Auteur inconnu"))

                with st.expander("📖 Voir détails"):
                    st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                    st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                    st.write(f"**Année :** {book.get('publication_year', 'Inconnue')}")
                    st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                    st.markdown("**Description :**")
                    st.write(book.get("description", "Pas de description disponible."))


# ---------------------------
# Recommandations SBERT
# ---------------------------
else:
    @st.cache_resource
    def load_sbert_model():
        path = MODELS_DIR / DIR / "sbert_model"
        return SentenceTransformer(str(path))

    @st.cache_data
    def load_embeddings_sbert():
        path = PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy"
        return np.load(path)
    
    sbert_model = load_sbert_model()
    embeddings_sbert = load_embeddings_sbert()

    nb_sugg = 10

    book_title = st.text_input("Titre du livre pour la recommandation Sentence-BERT")

    selected_title = None
    if book_title:
        suggestions_df = suggest_titles(book_title, tfidf, tfidf_matrix, content_df, k=nb_sugg)
        suggestion_list = suggestions_df['title'] + " - " + suggestions_df['authors']
        selected_title_author = st.selectbox("Titres suggérés :", suggestion_list)
        selected_title = selected_title_author.split(" - ")[0]

    if st.button("Rechercher S-BERT"):
        top_books = recommandation_content_top_k_gdr(selected_title, embeddings_sbert, sbert_model, content_df, top_k)

        cols = st.columns(5)
        for i, (_, book) in enumerate(top_books.iterrows()):
            col = cols[i % 5]
            with col:
                st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
                st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
                st.caption(book.get("authors", "Auteur inconnu"))

                with st.expander("📖 Voir détails"):
                    st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                    st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                    st.write(f"**Année :** {book.get('publication_year', 'Inconnue')}")
                    st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                    st.markdown("**Description :**")
                    st.write(book.get("description", "Pas de description disponible."))

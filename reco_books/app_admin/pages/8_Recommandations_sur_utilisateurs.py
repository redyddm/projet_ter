import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gensim
from recommandation_de_livres.iads.app_ui import display_book_card
from recommandation_de_livres.iads.collabo_utils import recommandation_collaborative_top_k
from recommandation_de_livres.iads.content_utils import (
    recommandation_content_top_k, suggest_titles, user_profile_embedding, recommandation_content_user_top_k
)
from recommandation_de_livres.iads.hybrid_utils import recommandation_hybride
from recommandation_de_livres.loaders.load_data import load_parquet, load_pkl, load_csv
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR

st.title("⚙️ Recommandations")

# -----------------------------
# Chargement des données
# -----------------------------
DIR = st.session_state['DIR']

@st.cache_data
def load_content():
    return load_parquet(PROCESSED_DATA_DIR / DIR / "content_dataset.parquet")

@st.cache_resource
def load_tfidf():
    with open(MODELS_DIR / DIR / "tfidf_model.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(PROCESSED_DATA_DIR / DIR / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    return tfidf, tfidf_matrix

@st.cache_resource
def load_svd_model():
    return load_pkl(MODELS_DIR / DIR / "svd_model.pkl")

@st.cache_resource
def load_w2v_model():
    return gensim.models.Word2Vec.load(str(MODELS_DIR / DIR / "word2vec.model"))

@st.cache_resource
def load_sbert_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(MODELS_DIR / DIR / "sbert_model"))

@st.cache_data
def load_embeddings_w2v():
    return np.load(PROCESSED_DATA_DIR / DIR / "embeddings_w2v.npy")

@st.cache_data
def load_embeddings_sbert():
    return np.load(PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy")

@st.cache_resource
def load_knn_w2v():
    import joblib
    path = MODELS_DIR / DIR / "knn_model_w2v.joblib"
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_knn_sbert():
    import joblib
    path = MODELS_DIR / DIR / "knn_model_sbert.joblib"
    return joblib.load(path) if path.exists() else None

# -----------------------------
# Initialisation
# -----------------------------
if "books" not in st.session_state:
    st.session_state["books"] = load_content()
if "ratings" not in st.session_state:
    st.session_state["ratings"] = load_parquet(PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet")

books = st.session_state["books"]
ratings = st.session_state["ratings"]
tfidf, tfidf_matrix = load_tfidf()

# -----------------------------
# Choix mode : utilisateur ou titre
# -----------------------------
reco_mode = st.radio(
    "Générer recommandations pour :",
    ["Utilisateur", "Titre de départ"]
)

selected_user = None
selected_title = None

if reco_mode == "Utilisateur":
    user_list = ratings["user_id"].unique()
    selected_user = st.selectbox("Sélectionner un utilisateur", user_list)
elif reco_mode == "Titre de départ":
    book_title_input = st.text_input("Titre du livre de départ")
    if book_title_input:
        suggestions_df = suggest_titles(book_title_input, tfidf, tfidf_matrix, books, k=10)
        suggestion_list = suggestions_df['title'] + " - " + suggestions_df['authors']
        selected_title_author = st.selectbox("Titres suggérés :", suggestion_list)
        selected_title = selected_title_author.split(" - ")[0]

# -----------------------------
# Choix modèle
# -----------------------------
model_type = st.selectbox(
    "Choisir le modèle",
    ["Word2Vec", "Sentence-BERT", "SVD", "Hybride"]
)
top_k = st.slider("Nombre de recommandations", 1, 20, 5)
alpha = None
if model_type == "Hybride":
    alpha = st.slider("Pondération collaborative vs contenu (alpha)", 0.0, 1.0, 0.5, 0.05)
    content_model_type = st.selectbox("Choisir le modèle de contenu", ["Word2Vec", "Sentence-BERT"])

# -----------------------------
# Lancer recommandation
# -----------------------------
if st.button("Lancer la recommandation"):

    embeddings = None
    content_model = None
    knn = None

    # Si utilisateur et profil disponible
    if reco_mode == "Utilisateur" and selected_user is not None:
        for m in ["Word2Vec", "Sentence-BERT"]:
            if model_type == m:
                embeddings = load_embeddings_w2v() if m=="Word2Vec" else load_embeddings_sbert()
                content_model = load_w2v_model() if m=="Word2Vec" else load_sbert_model()
                knn = load_knn_w2v() if m=="Word2Vec" else load_knn_sbert()

        if model_type in ["Word2Vec", "Sentence-BERT"]:
            item_id_to_idx = {item_id: idx for idx, item_id in enumerate(books['item_id'])}
            user_vec = user_profile_embedding(selected_user, ratings, embeddings, item_id_to_idx)
            if user_vec is not None:
                top_books, _ = recommandation_content_user_top_k(
                    selected_user, embeddings, content_model, books, ratings, knn=knn, k=top_k
                )
            else:
                st.warning("L'utilisateur n'a pas encore de notes. Veuillez saisir un titre de départ.")
                reco_mode = "Titre de départ"

    if reco_mode == "Titre de départ" and selected_title:
        if model_type in ["Word2Vec", "Sentence-BERT"]:
            embeddings = load_embeddings_w2v() if model_type=="Word2Vec" else load_embeddings_sbert()
            content_model = load_w2v_model() if model_type=="Word2Vec" else load_sbert_model()
            knn = load_knn_w2v() if model_type=="Word2Vec" else load_knn_sbert()
            top_books, _ = recommandation_content_top_k(
                selected_title, embeddings, content_model, books, knn=knn, k=top_k
            )

    elif model_type == "SVD" and selected_user is not None:
        svd_model = load_svd_model()
        top_books, _ = recommandation_collaborative_top_k(
            top_k, selected_user, svd_model, ratings, books
        )
    elif model_type == "Hybride":
        embeddings = load_embeddings_w2v() if content_model_type=="Word2Vec" else load_embeddings_sbert()
        content_model = load_w2v_model() if content_model_type=="Word2Vec" else load_sbert_model()
        knn = load_knn_w2v() if content_model_type=="Word2Vec" else load_knn_sbert()
        svd_model = load_svd_model()
        top_books = recommandation_hybride(
            user_id=selected_user,
            collaborative_model=svd_model,
            content_model=content_model,
            content_df=books,
            collaborative_df=ratings,
            books=books,
            embeddings=embeddings,
            alpha=alpha,
            knn=knn,
            k=top_k,
            top_k_content=50
        )

    # -----------------------------
    # Merge et affichage
    # -----------------------------
    if top_books is not None and not top_books.empty:
        cols_to_keep = [c for c in books.columns if c not in top_books.columns]
        top_books_full = top_books.merge(books[cols_to_keep + ["item_id"]], on="item_id", how="left") if cols_to_keep else top_books.copy()
        
        cols = st.columns(5)
        for i, (_, book) in enumerate(top_books_full.iterrows()):
            col = cols[i % 5]
            with col:
                display_book_card(book, allow_add=False, page_context="admin", show_rating_type="predicted")
    else:
        st.warning("Aucune recommandation trouvée.")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gensim
from recommandation_de_livres.iads.app_ui import display_book_card
from recommandation_de_livres.iads.collabo_utils import recommandation_collaborative_top_k
from recommandation_de_livres.iads.content_utils import recommandation_content_top_k, suggest_titles
from recommandation_de_livres.iads.hybrid_utils import recommandation_hybride
from recommandation_de_livres.loaders.load_data import load_parquet, load_pkl, load_csv
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR

st.title("⚙️ Admin - Recommandations")

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
    path = MODELS_DIR / DIR / "svd_model.pkl"
    return load_pkl(path)

@st.cache_resource
def load_w2v_model():
    path = MODELS_DIR / DIR / "word2vec.model"
    return gensim.models.Word2Vec.load(str(path))

@st.cache_resource
def load_sbert_model():
    from sentence_transformers import SentenceTransformer
    path = MODELS_DIR / DIR / "sbert_model"
    return SentenceTransformer(str(path))

@st.cache_data
def load_embeddings_w2v():
    return np.load(PROCESSED_DATA_DIR / DIR / "embeddings_w2v.npy")

@st.cache_data
def load_embeddings_sbert():
    return np.load(PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy")

# --- NOUVEAU : Chargement KNN Goodreads ---
@st.cache_resource
def load_knn_sbert():
    path = MODELS_DIR / DIR / "knn_model_sbert.joblib"
    if path.exists():
        import joblib
        return joblib.load(path)
    return None  # si pas trouvé, on retourne None

@st.cache_resource
def load_knn_w2v():
    path = MODELS_DIR / DIR / "knn_model_w2v.joblib"
    if path.exists():
        import joblib
        return joblib.load(path)
    return None  # si pas trouvé, on retourne None

if "books" not in st.session_state:
    st.session_state["books"] = load_content()
if "ratings" not in st.session_state:
    st.session_state["ratings"] = load_parquet(PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet")
if "users" not in st.session_state:
    st.session_state["users"] = load_csv(PROCESSED_DATA_DIR / DIR / "users.csv")

books = st.session_state["books"]
ratings = st.session_state["ratings"]
tfidf, tfidf_matrix = load_tfidf()
# -----------------------------
# Choix utilisateur
# -----------------------------
user_list = ratings["user_id"].unique()
selected_user = st.selectbox("Sélectionner un utilisateur", user_list)

# -----------------------------
# Choix type de modèle
# -----------------------------
model_type = st.selectbox(
    "Choisir le modèle",
    ["SVD", "KNN Collaborative", "Word2Vec", "Sentence-BERT", "Hybride"]
)

# -----------------------------
# Paramètres techniques
# -----------------------------
top_k = st.slider("Nombre de recommandations", 1, 20, 5)

selected_title = None
if model_type in ["Word2Vec", "Sentence-BERT"]:
    book_title_input = st.text_input("Titre du livre de départ")
    if book_title_input:
        suggestions_df = suggest_titles(book_title_input, tfidf, tfidf_matrix, books, k=10)
        suggestion_list = suggestions_df['title'] + " - " + suggestions_df['authors']
        selected_title_author = st.selectbox("Titres suggérés :", suggestion_list)
        selected_title = selected_title_author.split(" - ")[0]

alpha = None
if model_type == "Hybride":
    alpha = st.slider("Pondération collaborative vs contenu (alpha)", 0.0, 1.0, 0.5, 0.05)
    content_model_type = st.selectbox(
        "Choisir le modèle de contenu",
        ["Sentence-BERT", "Word2Vec"]
    )

# -----------------------------
# Lancer la recommandation
# -----------------------------


if st.button("Lancer la recommandation"):
    
    if model_type == "SVD":
        svd_model=load_svd_model()
        top_books, scores = recommandation_collaborative_top_k(
            k=top_k,
            user_id=selected_user,
            model=svd_model,
            ratings=ratings,
            books=books
        )
    elif model_type == "Word2Vec":
        embeddings_w2v=load_embeddings_w2v()
        w2v_model=load_w2v_model()
        knn_w2v=load_knn_w2v()
        top_books, _ = recommandation_content_top_k(
            selected_title,
            embeddings_w2v,
            w2v_model,
            books,
            knn=knn_w2v,  # <- utilisation KNN ici
            k=top_k
        )
    elif model_type == "Sentence-BERT":
        embeddings_sbert=load_embeddings_sbert()
        sbert_model=load_sbert_model()
        knn_sbert=load_knn_sbert()
        top_books, _ = recommandation_content_top_k(
            selected_title,
            embeddings_sbert,
            sbert_model,
            books,
            knn=knn_sbert,  # <- utilisation KNN ici
            k=top_k
        )
    elif model_type == "Hybride":
        
        # --- Choix du modèle de contenu ---    
        content_model = None
        embeddings = None
        knn = None

        if content_model_type == "Sentence-BERT":
            embeddings = load_embeddings_sbert()
            content_model = load_sbert_model()
            knn = load_knn_sbert()
        elif content_model_type == "Word2Vec":
            embeddings = load_embeddings_w2v()
            content_model = load_w2v_model()
            knn = load_knn_w2v()

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
            k=top_k,
            top_k_content=50,
            knn=knn
        )

    st.dataframe(top_books)
    
    # Merge pour récupérer colonnes infos
    cols_to_keep = [c for c in books.columns if c not in top_books.columns]
    top_books_full = top_books.merge(books[cols_to_keep + ["item_id"]], on="item_id", how="left")
    
    # Affichage
    cols = st.columns(5)
    for i, (_, book) in enumerate(top_books_full.iterrows()):
        col = cols[i % 5]
        with col:
            display_book_card(book, allow_add=False, page_context="admin", show_rating_type="predicted")

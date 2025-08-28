import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gensim
from pathlib import Path
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.app_ui import display_book_card, stars
from recommandation_de_livres.loaders.load_data import load_pkl, load_parquet
from recommandation_de_livres.iads.collabo_utils import recommandation_collaborative_top_k
from recommandation_de_livres.iads.content_utils import suggest_titles, recommandation_content_top_k, user_profile_embedding, recommandation_content_user_top_k
from recommandation_de_livres.iads.hybrid_utils import recommandation_hybride

DIR = st.session_state['DIR']

# ---------------------------
# Vérifier connexion
# ---------------------------
if not st.session_state.get("logged_in", False):
    st.warning("🚪 Veuillez vous connecter pour accéder à cette page.")
    st.stop()

books = st.session_state["books"]
ratings = st.session_state["ratings"]
if "ratings_count" not in books.columns:
    rating_count = ratings.groupby('item_id')['rating'].count().to_frame(name='ratings_count').reset_index()
    books = books.merge(rating_count, on='item_id', how='left')

    st.session_state["books"]=books


# ---------------------------
# Chargement ressources
# ---------------------------
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
def load_sbert_model():
    from sentence_transformers import SentenceTransformer
    path = MODELS_DIR / DIR / "sbert_model"
    return SentenceTransformer(str(path))

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

tfidf, tfidf_matrix = load_tfidf()
content_df = load_content()

# ---------------------------
# Formulaire unique
# ---------------------------
reco_type = st.selectbox(
    "Type de recommandation",
    ["Recommandations basées sur vos goûts", "Livres proches en thème et style", "Recommandations personnalisées"]
)
top_k = st.slider("Nombre de recommandations", 1, 20, 5)

user_id=st.session_state["user_id"]
if str(user_id) not in str(ratings['user_id']):
    if reco_type in ["Livres proches en thème et style"]:
        book_title_input = st.text_input("Titre du livre de départ")
        if book_title_input:
            suggestions_df = suggest_titles(book_title_input, tfidf, tfidf_matrix, content_df, k=10)
            suggestion_list = suggestions_df['title'] + " - " + suggestions_df['authors']
            selected_title_author = st.selectbox("Titres suggérés :", suggestion_list)
            selected_title = selected_title_author.split(" - ")[0]

selected_title = None

# Slider alpha pour hybride
if reco_type == "Recommandations personnalisées":
    alpha = st.slider(
        "Influence des types de recommandations",
        0.0, 1.0, 0.5, 0.05,
        help="0 = uniquement proches en thèmes et styles, 1 = uniquement basé sur vos goûts"
    )

# ---------------------------
# Recherche et affichage
# ---------------------------
# ---------------------------
# Recherche et affichage
# ---------------------------
if st.button("Rechercher"):

    # Choix du modèle et embeddings selon reco_type
    if reco_type == "Recommandations basées sur vos goûts":
        svd_model = load_svd_model()
        top_books, _ = recommandation_collaborative_top_k(
            k=top_k,
            user_id=user_id,
            model=svd_model,
            ratings=ratings,
            books=books
        )

    elif reco_type in ["Livres proches en thème et style"]:
        embeddings = load_embeddings_sbert()
        knn = load_knn_sbert()

        # Vérifier si l'utilisateur a un profil
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(books['item_id'])}
        user_vec = user_profile_embedding(user_id, ratings, embeddings, item_id_to_idx)

        if user_vec is not None:
            # Reco basée sur profil utilisateur
            top_books, _ = recommandation_content_user_top_k(
                st.session_state["user_id"],
                embeddings,
                content_df,
                ratings,
                knn=knn,
                k=top_k
            )

            if selected_title:
                model = load_sbert_model()
                top_books, _ = recommandation_content_top_k(
                    selected_title,
                    embeddings,
                    None,
                    content_df,
                    knn=knn,
                    k=top_k
                )

    elif reco_type == "Recommandations personnalisées":
        embeddings = load_embeddings_sbert()
        sbert_model = load_sbert_model()
        knn_sbert = load_knn_sbert()
        svd_model = load_svd_model()

        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(books['item_id'])}
        user_vec = user_profile_embedding(user_id, ratings, embeddings, item_id_to_idx)

        if user_vec is not None:
            # Reco basée sur profil utilisateur            
            top_books = recommandation_hybride(
                user_id=st.session_state["user_id"],
                collaborative_model=svd_model,
                content_model=None,
                content_df=content_df,
                collaborative_df=ratings,
                books=books,
                embeddings=embeddings,
                alpha=alpha,
                knn=knn_sbert,
                k=top_k,
                top_k_content=100
            )
        else:
            st.warning("Vous n'avez aucun livre dans votre collection. Veuillez en ajouter afin de pouvoir recevoir des recommandations personnalisées.")
            st.stop()

    # Merge pour récupérer les colonnes nécessaires comme average_rating
    top_books['item_id'] = top_books['item_id'].astype(object)
    books['item_id'] = books['item_id'].astype(object)
    cols_to_keep = [c for c in books.columns if c not in top_books.columns]
    top_books_full = top_books.merge(
        books[cols_to_keep + ["item_id"]],
        on="item_id",
        how="left"
    )

    # Affichage
    cols = st.columns(5)
    for i, (_, book) in enumerate(top_books_full.iterrows()):
        col = cols[i % 5]
        with col:
            display_book_card(book, allow_add=True, page_context="reco", show_rating_type="predicted")

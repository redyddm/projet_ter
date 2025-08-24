import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gensim
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.utils import stars
from recommandation_de_livres.loaders.load_data import load_pkl, load_parquet
from recommandation_de_livres.iads.collabo_utils import recommandation_collaborative_top_k
from recommandation_de_livres.iads.content_utils import suggest_titles, recommandation_content_top_k
from recommandation_de_livres.iads.hybrid_utils import recommandation_hybride_vectorisee
from tqdm import tqdm

DIR = st.session_state['DIR']

# ---------------------------
# Vérifier connexion
# ---------------------------
if not st.session_state.get("logged_in", False):
    st.warning("🚪 Veuillez vous connecter pour accéder à cette page.")
    st.stop()

books = st.session_state["books"]
ratings = st.session_state["ratings"]
users = st.session_state["users"]

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

tfidf, tfidf_matrix = load_tfidf()
content_df = load_content()
svd_model = load_svd_model()
w2v_model = load_w2v_model()
embeddings_w2v = load_embeddings_w2v()
sbert_model = load_sbert_model()
embeddings_sbert = load_embeddings_sbert()

# ---------------------------
# Formulaire unique
# ---------------------------
reco_type = st.selectbox(
    "Type de recommandation",
    ["Recommandations basées sur vos goûts", "Livres similaires à celui-ci", "Livres proches en thème et style", "Recommandations personnalisées"]
)
top_k = st.slider("Nombre de recommandations", 1, 20, 5)

selected_title = None
if reco_type in ["Livres similaires à celui-ci", "Livres proches en thème et style"]:
    book_title_input = st.text_input("Titre du livre de départ")
    if book_title_input:
        suggestions_df = suggest_titles(book_title_input, tfidf, tfidf_matrix, content_df, k=10)
        suggestion_list = suggestions_df['title'] + " - " + suggestions_df['authors']
        selected_title_author = st.selectbox("Titres suggérés :", suggestion_list)
        selected_title = selected_title_author.split(" - ")[0]

# Slider alpha pour hybride
if reco_type == "Recommandations personnalisées":
    alpha = st.slider(
        "Pondération Collaborative vs Contenu (alpha)",
        0.0, 1.0, 0.5, 0.05,
        help="0 = uniquement contenu, 1 = uniquement collaboratif"
    )

# ---------------------------
# Recherche et affichage
# ---------------------------
if st.button("Rechercher"):

    # Choix du modèle et embeddings selon reco_type
    if reco_type == "Recommandations basées sur vos goûts":
        top_books, top_k_rating = recommandation_collaborative_top_k(
            k=top_k,
            user_id=st.session_state["user_id"],
            model=svd_model,
            ratings=ratings,
            books=books
        )
    elif reco_type == "Livres similaires à celui-ci":
        top_books, sim = recommandation_content_top_k(selected_title, embeddings_w2v, w2v_model, content_df, top_k)
    elif reco_type == "Livres proches en thème et style":
        top_books, sim = recommandation_content_top_k(selected_title, embeddings_sbert, sbert_model, content_df, top_k)
    elif reco_type == "Recommandations personnalisées":
        top_books = recommandation_hybride_vectorisee(
            user_id=st.session_state["user_id"],
            collaborative_model=svd_model,
            content_model=sbert_model,
            content_df=content_df,
            collaborative_df=ratings,
            books=books,
            embeddings=embeddings_sbert,
            alpha=alpha,
            k=top_k,
            top_k_content=30
        )

    if top_books is not None and not top_books.empty:
        cols = st.columns(5)
        for i, (_, book) in enumerate(top_books.iterrows()):
            col = cols[i % 5]
            with col:
                st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
                st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
                # Affiche score hybride si hybride, sinon prédiction SVD
                score_display = book.get("score_hybride") if "score_hybride" in book else book.get("predicted_rating", 0)
                st.markdown(stars(score_display))
                st.caption(book.get("authors", "Auteur inconnu"))

                with st.expander("📖 Voir détails"):
                    st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                    st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                    st.write(f"**Année :** {book.get('publication_year', book.get('year', 'Inconnue'))}")
                    st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                    st.markdown("**Description :**")
                    st.write(book.get("description", "Pas de description disponible."))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.app_ui import display_book_card
from recommandation_de_livres.loaders.load_data import load_parquet

# -----------------------------
# Chargement données
# -----------------------------
DIR = st.session_state['DIR']

@st.cache_data
def load_content():
    return load_parquet(PROCESSED_DATA_DIR / DIR / "content_dataset.parquet")
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
if "embeddings_sbert" not in st.session_state:
    st.session_state["embeddings_sbert"] = load_embeddings_sbert()

books = st.session_state['books']  # DataFrame avec 'item_id', 'title', 'authors', etc.
ratings = st.session_state['ratings']  # DataFrame avec 'user_id', 'item_id', 'rating'
embeddings = st.session_state['embeddings_sbert']  # np.array, indexé par item_id

item_id_to_idx = {item_id: idx for idx, item_id in enumerate(books['item_id'])}

# -----------------------------
# Sélection utilisateur
# -----------------------------
user_list = ratings['user_id'].unique()
selected_user = st.selectbox("Sélectionner un utilisateur", user_list)

top_k = st.slider("Nombre de recommandations", 1, 20, 5)

# -----------------------------
# Calcul du profil utilisateur
# -----------------------------
def user_profile_embedding(user_id, ratings, embeddings, item_id_to_idx):
    rated_books = ratings[ratings["user_id"] == user_id]
    if rated_books.empty:
        return None

    vectors = []
    for _, row in rated_books.iterrows():
        idx = item_id_to_idx.get(row['item_id'])
        if idx is not None:
            vectors.append(embeddings[idx] * row['rating'])  # pondération par la note

    if not vectors:
        return None

    user_vec = np.mean(vectors, axis=0)
    return user_vec

user_vec = user_profile_embedding(selected_user, ratings, embeddings, item_id_to_idx)

if user_vec is None:
    st.warning("Cet utilisateur n'a pas encore de notes pour calculer un profil.")
    st.stop()

# -----------------------------
# Calcul similarité cosinus
# -----------------------------
all_embeddings = embeddings  # np.array des embeddings tous livres
similarities = cosine_similarity(user_vec.reshape(1, -1), all_embeddings).flatten()

# Exclure les livres déjà notés
rated_ids = set(ratings[ratings["user_id"] == selected_user]["item_id"])
candidates = [(i, sim) for i, sim in enumerate(similarities) if books.loc[i, "item_id"] not in rated_ids]

# Top-K
top_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
top_books_idx = [i for i, _ in top_candidates]
top_books = books.iloc[top_books_idx]

# -----------------------------
# Affichage
# -----------------------------
st.subheader(f"Top-{top_k} recommandations basées sur contenu pour {selected_user}")
cols = st.columns(5)
for i, (_, book) in enumerate(top_books.iterrows()):
    with cols[i % 5]:
        display_book_card(book, allow_add=False, page_context="admin", show_rating_type="average")

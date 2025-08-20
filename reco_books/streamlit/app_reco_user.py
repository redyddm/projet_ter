import streamlit as st
import pandas as pd
import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR
from recommandation_de_livres.iads.utils import stars
from recommandation_de_livres.loaders.load_data import load_csv, load_pkl
from recommandation_de_livres.iads.svd_utils_gdr import recommandation_collaborative_top_k_gdr

DIR = 'goodreads'

# Chargement des datasets
@st.cache_data
def load_user_ratings():
    return pd.read_csv(PROCESSED_DATA_DIR / DIR / "collaborative_dataset.csv")

@st.cache_data
def load_books():
    return pd.read_csv(INTERIM_DATA_DIR / DIR / "books_authors.csv")

@st.cache_resource
def load_svd_model():
    return load_pkl(MODELS_DIR / DIR / "svd_model.pkl")

user_ratings = load_user_ratings()
books = load_books()
model = load_svd_model()

# --- Connexion utilisateur ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None

if not st.session_state["logged_in"]:
    user_index_input = st.selectbox(
        "Sélectionnez votre user_id",
        sorted(user_ratings["user_index"].unique())
    )
    if st.button("Se connecter"):
        if user_index_input in user_ratings["user_index"].values:
            st.session_state["logged_in"] = True
            st.session_state["user_index"] = user_index_input
            st.session_state["user_id"] = user_ratings.loc[user_ratings["user_index"] == user_index_input, "user_id"].iloc[0]
        else:
            st.error("Utilisateur non trouvé")

# --- Sidebar navigation ---
if st.session_state.get("logged_in", False):
    st.sidebar.write(f"✅ Connecté : user {st.session_state['user_index']}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state["logged_in"] = False
        st.session_state["user_id"] = None
        st.session_state["user_index"] = None

    page = st.sidebar.radio("Navigation", ["📚 Ma Bibliothèque", "⭐ Recommandations"])

    # --- Page Bibliothèque ---
    if page == "📚 Ma Bibliothèque":
        books_user = user_ratings[user_ratings["user_index"] == st.session_state["user_index"]]
        if not books_user.empty:
            st.subheader("📚 Ma Bibliothèque")
            cols = st.columns(3)
            for i, (_, book) in enumerate(books_user.iterrows()):
                col = cols[i % 3]
                with col:
                    st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
                    st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
                    st.markdown(stars(book.get("rating", 0)))
                    st.caption(book.get("authors", "Auteur inconnu"))
                    with st.expander("📖 Voir détails"):
                        st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                        st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                        st.write(f"**Année :** {book.get('year', 'Inconnue')}")
                        st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                        st.markdown("**Description :**")
                        st.write(book.get("description", "Pas de description disponible."))
        else:
            st.subheader("📚 Ma Bibliothèque")
            st.write("Aucun livre trouvé pour cet utilisateur")

    # --- Page Recommandations SVD ---
    # --- Page Recommandations SVD ---
    elif page == "⭐ Recommandations":
        st.subheader("⭐ Recommandations pour vous")
        top_k = st.slider("Nombre de recommandations souhaité", min_value=1, max_value=20, value=5)
        
        if st.button("Rechercher"):
            top_books = recommandation_collaborative_top_k_gdr(
                k=top_k,
                user_id=st.session_state["user_id"],
                model=model,
                ratings=user_ratings
            )

            # Créer une grille 3 colonnes
            cols = st.columns(3)
            for i, (_, book) in enumerate(top_books.iterrows()):
                col = cols[i % 3]
                with col:
                    # Affichage image + titre + étoiles
                    st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
                    st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
                    st.markdown(stars(book.get("predicted_rating", 0)))  # note prédite
                    st.caption(book.get("authors", "Auteur inconnu"))

                    # Détails en expander
                    with st.expander("📖 Voir détails"):
                        st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                        st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                        st.write(f"**Année :** {book.get('year', 'Inconnue')}")
                        st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                        st.markdown("**Description :**")
                        st.write(book.get("description", "Pas de description disponible."))

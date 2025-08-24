import streamlit as st
import pandas as pd
from pathlib import Path

from recommandation_de_livres.loaders.load_data import load_csv
from recommandation_de_livres.config import INTERIM_DATA_DIR, RAW_DATA_DIR

# ---------------------------
# Sécurité : Vérifier que DIR est bien défini dans la session
# ---------------------------
if 'DIR' not in st.session_state:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à la page d'accueil pour en choisir un.")
    st.stop()

DIR = st.session_state['DIR']

# ---------------------------
# Vérification de l'existence des fichiers RAW
# ---------------------------
book_path = RAW_DATA_DIR / DIR / "books.csv"
rating_path = RAW_DATA_DIR / DIR / "ratings.csv"

if not book_path.exists() or not rating_path.exists():
    st.error(f"Impossible de trouver `books.csv` ou `ratings.csv` dans {RAW_DATA_DIR / DIR}")
    st.stop()

# ---------------------------
# PAGE PRINCIPALE
# ---------------------------
st.set_page_config(page_title=f"Uniformisation des colonnes - {DIR}", layout="wide", page_icon="📚")
st.title(f"📚 Uniformisation des colonnes - {DIR}")

@st.cache_data
def load_books(path): return load_csv(path)

@st.cache_data
def load_ratings(path): return load_csv(path)

# ---------------------------
# Sidebar : choix du type de dataset
# ---------------------------
st.sidebar.title("Uniformisation")
dataset_choice = st.sidebar.selectbox("Choix du type de dataset", ["books", "ratings"])

# ---------------------------
# Chargement du dataset sélectionné
# ---------------------------
df = load_books(book_path) if dataset_choice == "books" else load_ratings(rating_path)

st.subheader(f"Aperçu du dataset `{dataset_choice}`")
st.dataframe(df.head())

# ======================================================================
# UNIFORMISATION DES RATINGS
# ======================================================================
if dataset_choice == "ratings":
    st.subheader("🛠️ Mapping des colonnes pour uniformisation (Ratings)")
    user_col = st.selectbox("Colonne utilisateur (user_id)", ["None"] + list(df.columns), key="user_col")
    item_col = st.selectbox("Colonne item/livre (item_id)", ["None"] + list(df.columns), key="item_col")
    rating_col = st.selectbox("Colonne note (rating)", ["None"] + list(df.columns), key="rating_col")

    if st.button("✅ Valider et uniformiser Ratings", key="validate_ratings"):
        standardized_df = pd.DataFrame()

        for col, name in zip([user_col, item_col, rating_col], ["user_id", "item_id", "rating"]):
            if col != "None":
                standardized_df[name] = df[col]
            else:
                st.warning(f"La colonne {name} n'a pas été sélectionnée.")

        standardized_df = standardized_df.dropna(subset=["user_id", "item_id", "rating"])
        standardized_df["rating"] = pd.to_numeric(standardized_df["rating"], errors="coerce")
        standardized_df = standardized_df.dropna(subset=["rating"])

        st.success("Dataset `ratings` uniformisé avec succès !")
        st.dataframe(standardized_df.head())

        save_path = INTERIM_DATA_DIR / DIR / "ratings_uniform.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        standardized_df.to_csv(save_path, index=False)
        st.info(f"Dataset sauvegardé sous : `{save_path}`")

        st.download_button(
            "⬇️ Télécharger le dataset uniformisé (ratings)",
            data=standardized_df.to_csv(index=False).encode('utf-8'),
            file_name="ratings_uniform.csv",
            mime="text/csv"
        )

# ======================================================================
# UNIFORMISATION DES BOOKS
# ======================================================================
elif dataset_choice == "books":
    st.subheader("🛠️ Mapping des colonnes pour uniformisation")
    item_col = st.selectbox("Colonne item/livre (item_id)", ["None"] + list(df.columns), key="book_item_col")
    title_col = st.selectbox("Titre du livre", ["None"] + list(df.columns), key="book_title_col")
    author_col = st.selectbox("Auteur", ["None"] + list(df.columns), key="book_author_col")
    publisher_col = st.selectbox("Éditeur", ["None"] + list(df.columns), key="book_publisher_col")
    year_col = st.selectbox("Année de publication", ["None"] + list(df.columns), key="book_year_col")
    image_col = st.selectbox("Image URL", ["None"] + list(df.columns), key="book_image_col")

    if st.button("✅ Valider et uniformiser Books", key="validate_books"):
        standardized_df = pd.DataFrame()

        for col, name in zip([item_col, title_col, author_col], ["item_id", "title", "authors"]):
            if col != "None":
                standardized_df[name] = df[col]
            else:
                st.warning(f"La colonne {name} n'a pas été sélectionnée.")

        standardized_df = standardized_df.dropna(subset=["item_id", "title"])

        st.success("Dataset `books` uniformisé avec succès !")
        st.dataframe(standardized_df.head())

        save_path = INTERIM_DATA_DIR / DIR / "books_uniform.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        standardized_df.to_csv(save_path, index=False)
        st.info(f"Dataset sauvegardé sous : `{save_path}`")

        st.download_button(
            "⬇️ Télécharger le dataset uniformisé (books)",
            data=standardized_df.to_csv(index=False).encode('utf-8'),
            file_name="books_uniform.csv",
            mime="text/csv"
        )

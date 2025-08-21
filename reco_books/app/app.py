import streamlit as st
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR, INTERIM_DATA_DIR

st.set_page_config(page_title="Accueil", layout="wide")

st.set_page_config(page_title="Accueil", page_icon="📚")

@st.cache_data
def load_books():
    return load_parquet(INTERIM_DATA_DIR / "goodreads/books_authors.parquet")

@st.cache_data
def load_ratings():
    return load_parquet(PROCESSED_DATA_DIR / "goodreads/collaborative_dataset.parquet")

if "books" not in st.session_state:
    st.session_state["books"] = load_books()

if "ratings" not in st.session_state:
    st.session_state["ratings"] = load_ratings()

st.title("📚 Bienvenue dans l'application de recommandations de livres")

st.write("""
Cette application vous permet :
- d'explorer vos données de livres,
- d'obtenir des recommandations personnalisées.
""")

st.info("👉 Utilisez le menu de gauche pour naviguer entre les pages.")
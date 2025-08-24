import streamlit as st
import pandas as pd
from pathlib import Path

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.loaders.load_data import load_parquet, load_csv
from recommandation_de_livres.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

st.set_page_config(page_title="Accueil", layout="wide", page_icon="📚")

choice = st.selectbox("Choix du dataset :", ["Recommender", "Goodreads", "Personnel"], index=0)

@st.cache_data
def load_books(dir_name):
    return load_parquet(PROCESSED_DATA_DIR / dir_name / "content_dataset.parquet")

@st.cache_data
def load_ratings(dir_name):
    return load_parquet(PROCESSED_DATA_DIR / dir_name / "collaborative_dataset.parquet")

@st.cache_data
def load_users(dir_name):
    return load_csv(PROCESSED_DATA_DIR / dir_name / "users.csv")

# Déterminer le DIR en fonction du choix
if choice.startswith("Personnel"):
    DATA_DIR = Path("data/raw")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    book_path = DATA_DIR / "books_uniform.csv"
    rating_path = DATA_DIR / "ratings_uniform.csv"
else:
    if choice.startswith("Recommender"):
        DATA_DIR = "recommender"
    elif choice.startswith("Goodreads"):
        DATA_DIR = "goodreads"

# Charger les données après avoir déterminé DIR
if st.button("Charger les données"):
    if choice.startswith("Personnel"):
        st.session_state["books"] = pd.read_csv(book_path)
        st.session_state["ratings"] = pd.read_csv(rating_path)
    else:
        st.session_state["books"] = load_books(DATA_DIR)
        st.session_state["ratings"] = load_ratings(DATA_DIR)
        st.session_state["users"] = load_users(DATA_DIR)

    st.session_state["DIR"]=DATA_DIR

# ---------------------------
# Session state : init
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["user_id"] = None

# ---------------------------
# Sidebar : Connexion
# ---------------------------
st.sidebar.title("🔑 Connexion")

if not st.session_state["logged_in"]:
    username_input = st.sidebar.text_input("Nom d’utilisateur (userX)")
    if st.sidebar.button("Se connecter"):
        users = st.session_state["users"]
        if username_input in users["username"].values:
            user_row = users.loc[users["username"] == username_input].iloc[0]
            st.session_state["logged_in"] = True
            st.session_state["username"] = user_row["username"]
            st.session_state["user_id"] = user_row["user_id"]
            st.session_state["user_index"] = user_row["user_index"]
            st.sidebar.success(f"Bienvenue {user_row['username']} 👋")
        else:
            st.sidebar.error("Utilisateur inconnu.")
else:
    st.sidebar.success(f"✅ Connecté : {st.session_state['username']}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["user_id"] = None

# ---------------------------
# Interface principale
# ---------------------------
st.title("📚 Bienvenue dans l'application de recommandations de livres")

if not st.session_state["logged_in"]:
    st.info("👉 Connectez-vous depuis la barre latérale pour accéder à vos recommandations et à votre collection.")
else:
    st.success(f"Bonjour {st.session_state['username']} ! 🎉")
    st.write("""
    Vous pouvez maintenant accéder aux fonctionnalités suivantes depuis le menu de gauche :
    - 📖 Explorer vos livres
    - ⭐ Voir vos recommandations personnalisées
    - 📚 Gérer votre collection
    """)
    if "page_num" in st.session_state:
        st.session_state["page_num"] = 1
        st.session_state["prev_clicked"] = False
        st.session_state["next_clicked"] = False

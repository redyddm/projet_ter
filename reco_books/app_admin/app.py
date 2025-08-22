import streamlit as st
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.loaders.load_data import load_parquet, load_csv
from recommandation_de_livres.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

st.set_page_config(page_title="Accueil", layout="wide", page_icon="📚")

DIR ='goodreads'

@st.cache_data
def load_books():
    return load_parquet(INTERIM_DATA_DIR / DIR / "books_authors.parquet")

@st.cache_data
def load_ratings():
    return load_parquet(PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet")

@st.cache_data
def load_users():
    return load_csv(RAW_DATA_DIR / DIR / "users.csv")

# ---------------------------
# Session state : init
# ---------------------------
if "books" not in st.session_state:
    st.session_state["books"] = load_books()

if "ratings" not in st.session_state:
    st.session_state["ratings"] = load_ratings()

if "users" not in st.session_state:
    st.session_state["users"] = load_users()

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
        st.session_state["page_num"] = 0
        st.session_state["prev_clicked"] = False
        st.session_state["next_clicked"] = False

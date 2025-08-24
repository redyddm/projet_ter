import streamlit as st
import pandas as pd
from pathlib import Path
import sys, os

# Import modules internes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recommandation_de_livres.loaders.load_data import load_parquet, load_csv
from recommandation_de_livres.config import PROCESSED_DATA_DIR

# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(page_title="Accueil", layout="wide", page_icon="📚")

# ---------------------------
# Chargement automatique des données
# ---------------------------
DATA_DIR = "goodreads"  # <-- défini par toi en dur ou dans un fichier config

@st.cache_data
def load_books():
    return load_parquet(PROCESSED_DATA_DIR / DATA_DIR / "content_dataset.parquet")

@st.cache_data
def load_ratings():
    return load_parquet(PROCESSED_DATA_DIR / DATA_DIR / "collaborative_dataset.parquet")

@st.cache_data
def load_users():
    return load_csv(PROCESSED_DATA_DIR / DATA_DIR / "users.csv")


st.session_state["books"] = load_books()
st.session_state["ratings"] = load_ratings()
st.session_state["users"] = load_users()
st.session_state["DIR"] = DATA_DIR

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
            st.session_state.update({
                "logged_in": True,
                "username": user_row["username"],
                "user_id": user_row["user_id"],
                "user_index": user_row["user_index"]
            })
            st.sidebar.success(f"Bienvenue {user_row['username']} 👋")
        else:
            st.sidebar.error("Utilisateur inconnu.")
else:
    st.sidebar.success(f"✅ Connecté : {st.session_state['username']}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.update({"logged_in": False, "username": None, "user_id": None})

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
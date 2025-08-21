import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# Pour ton projet
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.config import DB_PARAMS

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Accueil", page_icon="📚", layout="wide")

# ---------------------------
# Connexion PostgreSQL
# ---------------------------
def get_engine():
    return create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}/{DB_PARAMS['dbname']}"
    )

engine = get_engine()

# ---------------------------
# Fonctions de chargement
# ---------------------------
@st.cache_data
def load_books():
    query = "SELECT * FROM books"
    return pd.read_sql(query, engine)

@st.cache_data
def load_ratings():
    query = "SELECT * FROM ratings"
    return pd.read_sql(query, engine)

@st.cache_data
def load_users():
    query = "SELECT * FROM users"
    return pd.read_sql(query, engine)

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

    if "page_num" in st.session_state:
        st.session_state["page_num"] = 0
        st.session_state["prev_clicked"] = False
        st.session_state["next_clicked"] = False

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

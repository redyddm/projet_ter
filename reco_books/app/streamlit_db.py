import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------
# Connexion à PostgreSQL
# ---------------------------
DB_PARAMS = {
    "user": "user",
    "password": "pwd",
    "host": "localhost",
    "dbname": "Library"
}

engine = create_engine(
    f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}/{DB_PARAMS['dbname']}"
)

# ---------------------------
# Fonctions pour charger les données
# ---------------------------
@st.cache_data
def load_books():
    return pd.read_sql("SELECT * FROM books", engine)

@st.cache_data
def load_users():
    return pd.read_sql("SELECT * FROM users", engine)

@st.cache_data
def load_ratings():
    return pd.read_sql("SELECT * FROM ratings", engine)

# ---------------------------
# Charger les données
# ---------------------------
books_df = load_books()
users_df = load_users()
ratings_df = load_ratings()

# ---------------------------
# Interface Streamlit
# ---------------------------
st.title("Application de recommandation de livres")

# Sélection utilisateur
user_index = st.selectbox("Sélectionnez un utilisateur", users_df['user_index'])

# Récupérer l'user_index
user_id = users_df.loc[users_df['user_index'] == user_index, 'user_id'].values[0]

# Livres notés par l'utilisateur
user_ratings = ratings_df[ratings_df['user_id'] == user_id]
st.subheader(f"Livres déjà notés par l'utilisateur {user_index}")
st.dataframe(user_ratings[['title', 'authors', 'rating']])


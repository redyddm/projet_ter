import streamlit as st
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.config import PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_csv, load_parquet

DATA_DIR = "recommender"

st.session_state['DIR']=DATA_DIR

# ---------------------------
# Chargement des données si pas déjà en session
# ---------------------------
if "books" not in st.session_state:
    st.session_state["books"] = load_parquet(PROCESSED_DATA_DIR / DATA_DIR / "content_dataset.parquet")
if "ratings" not in st.session_state:
    st.session_state["ratings"] = load_parquet(PROCESSED_DATA_DIR / DATA_DIR / "collaborative_dataset.parquet")
if "users" not in st.session_state:
    st.session_state["users"] = load_csv(PROCESSED_DATA_DIR / DATA_DIR / "users.csv")

# ---------------------------
# Initialisation session
# ---------------------------
for key in ["logged_in", "username", "user_id", "user_index"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state["logged_in"] = st.session_state["logged_in"] or False

# ---------------------------
# Sidebar : Connexion / Création / Déconnexion
# ---------------------------
st.sidebar.title("🔑 Utilisateur")

if not st.session_state["logged_in"]:
    # ----------------- Connexion -----------------
    st.sidebar.subheader("Se connecter")
    username_input = st.sidebar.text_input("Nom d’utilisateur", key="login_username")
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
            st.rerun()  # Recharger la page immédiatement pour cacher le formulaire
        else:
            st.sidebar.error("Utilisateur inconnu.")

    # --------------- Création utilisateur ---------------
    st.sidebar.subheader("Créer un nouvel utilisateur")
    with st.sidebar.form("create_user_form"):
        new_username = st.text_input("Nom d’utilisateur", key="create_username")
        submitted = st.form_submit_button("Créer l'utilisateur")
        if submitted:
            users_df = st.session_state["users"]
            if new_username in users_df["username"].values:
                st.sidebar.error("Ce nom d’utilisateur existe déjà.")
            else:
                existing_indices = users_df["user_index"].tolist()
                new_user_index = max(existing_indices) + 1 if existing_indices else 0
                new_user_id = f"user_{new_user_index + 1}"

                new_user = pd.DataFrame([{
                    "user_id": new_user_id,
                    "username": new_username,
                    "user_index": new_user_index
                }])

                st.session_state["users"] = pd.concat([users_df, new_user], ignore_index=True)
                users_df_path = PROCESSED_DATA_DIR / DATA_DIR / "users.csv"
                st.session_state["users"].to_csv(users_df_path, index=False)
                st.sidebar.success(f"Utilisateur {new_username} créé ! Connectez-vous maintenant.")

else:
    # ----------------- Déconnexion -----------------
    st.sidebar.success(f"Connecté en tant que {st.session_state['username']} 👋")
    if st.sidebar.button("Se déconnecter"):
        for key in ["logged_in", "username", "user_id", "user_index"]:
            st.session_state[key] = None
        st.rerun()  # Recharger la page pour afficher les formulaires de connexion/création


# ---------------------------
# Interface principale
# ---------------------------
st.title("📚 Bienvenue dans l'application de recommandations de livres")

if not st.session_state["logged_in"]:
    st.info("👉 Connectez-vous depuis la barre latérale pour accéder à vos recommandations et à votre collection.")
else:
    username = st.session_state["username"]
    user_index = st.session_state["user_index"]
    st.success(f"Bonjour {username} ! 🎉")

    st.write("""
    Vous pouvez maintenant accéder aux fonctionnalités suivantes depuis le menu de gauche :
    - 📖 Consulter la bibliothèque
    - 📚 Gérer votre collection
    - ⭐ Voir vos recommandations personnalisées
    """)
    if "page_num" in st.session_state:
        st.session_state["page_num"] = 1
        st.session_state["prev_clicked"] = False
        st.session_state["next_clicked"] = False
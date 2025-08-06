import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import glob
import pickle 

# Tes fonctions de recommandation
from svd_recommandation import recommandation_collaborative_top_k, predict_unrated_books, get_unrated_item

# === Chargement des données et modèle ===
import streamlit as st
import pandas as pd
import glob
import re
import pickle

# 🔁 Cacher les fichiers chargés (dataframes)
@st.cache_data
def load_books():
    chemin = "./datasets/goodreads/books/*.csv"
    fichiers_csv = glob.glob(chemin)

    def extraire_debut(fichier):
        match = re.search(r'book(\d+)(k)?-', fichier)
        if match:
            nombre = int(match.group(1))
            if match.group(2):
                nombre *= 1000
            return nombre
        return float('inf')

    fichiers_csv = sorted(fichiers_csv, key=extraire_debut)
    books = pd.concat([pd.read_csv(fichier) for fichier in fichiers_csv], ignore_index=True)
    return books

@st.cache_data
def load_ratings():
    return pd.read_csv('datasets/goodreads/reco/collaborative_dataset.csv')

# 🔁 Cacher le modèle chargé (objet lourd)
@st.cache_resource
def load_model():
    with open('./models/goodreads/svd/svd_best_model.pkl', 'rb') as f:
        return pickle.load(f)
    
@st.cache_data(show_spinner=False)
def cached_predict_unrated_books(user_id, _model, non_rated):
    return predict_unrated_books(user_id, _model, non_rated)




st.title("Recommandation de Livres - SVD")

books = load_books()
ratings = load_ratings()
model = load_model()

# === Sélection utilisateur ===
user_id = st.slider("Sélectionnez un utilisateur :", 1, ratings['ID'].max(), 1)

# === Top-k recommandations ===
k = st.number_input("Nombre de recommandations", min_value=1, max_value=20, value=5)

if st.button("Générer les recommandations"):
    top_k_books = recommandation_collaborative_top_k(k, user_id, model, ratings, books)
    st.subheader(f"Top {k} livres recommandés pour l'utilisateur {user_id}")
    st.table(top_k_books['Name'])

    # Prédiction complète
    with st.spinner("Calcul des prédictions pour tous les livres non notés..."):
        non_rated = get_unrated_item(user_id, ratings)
        pred_df = cached_predict_unrated_books(user_id, model, non_rated)

    st.subheader("📊 Distribution des notes prédites")
    fig, ax = plt.subplots()
    sns.histplot(pred_df['note_predite'], bins=20, ax=ax)
    st.pyplot(fig)

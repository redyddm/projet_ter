import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.loaders.load_data import load_csv
from recommandation_de_livres.config import INTERIM_DATA_DIR

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Exploration des données", layout="wide", page_icon="📚")

@st.cache_data
def load_books(path):
    return load_csv(path)

@st.cache_data
def load_ratings(path):
    return load_csv(path)
# ---------------------------
# Interface principale
# ---------------------------
st.title("📚 Exploration des données")

# --- Choix du dataset global ---
choice = st.selectbox("Choix du dataset :", ["Recommender", "Goodreads", "Personnel"], index=0)

if choice.startswith("Personnel"):
    book_path = DATA_DIR / "books_uniform.csv"
    rating_path = DATA_DIR / "ratings_uniform.csv"
    
else:

    if choice.startswith("Recommender"):
        DIR = "recommender" 
    elif  choice.startswith("Goodreads"):
        DIR = "goodreads"

book_path = INTERIM_DATA_DIR / DIR / "books_uniform.csv"
rating_path = INTERIM_DATA_DIR / DIR / "ratings_uniform.csv"
# --- Sidebar ---
st.sidebar.title("Exploration EDA")
dataset_choice = st.sidebar.selectbox("Choix du type de dataset", ["books", "ratings"])

# --- Charger le dataset sélectionné ---
if dataset_choice == "books":
    df = load_books(book_path)
else:
    df = load_ratings(rating_path)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["Aperçu", "Top livres", "Top auteurs", "Distibution des notes"]
)

# --- Tab 1 : Aperçu ---
with tab1:
    st.subheader(f"Aperçu du dataset {dataset_choice}")
    st.dataframe(df.head())
    st.write("Dimensions :", df.shape)

    with st.expander("Info dataset"):
        info_df = pd.DataFrame({
            "Colonne": df.columns,
            "Type": df.dtypes.values,
            "Nombre de valeurs non-null": df.notnull().sum().values,
            "Nombre de valeurs null": df.isnull().sum().values,
            "Pourcentage de valeurs null": (df.isnull().mean() * 100).round(2).values
        })
        st.dataframe(info_df)

    with st.expander("Statistiques descriptives"):
        st.dataframe(df.describe())

# --- Tab 2 : Top livres ---
with tab2:
    st.subheader("Top 10 livres populaires avec stats des ratings")

    if dataset_choice == "ratings":
        # Charger books
        books_df = load_books(book_path)

        # Séparer ratings > 0 et ratings = 0
        ratings_positive = df[df["rating"] > 0]
        ratings_zero = df[df["rating"] == 0]

        # Nombre de ratings > 0 par livre
        count_positive = ratings_positive.groupby("item_id")["rating"].count().reset_index()
        count_positive.rename(columns={"rating": "num_ratings_positive"}, inplace=True)

        # Moyenne des ratings > 0
        avg_rating = ratings_positive.groupby("item_id")["rating"].mean().reset_index()
        avg_rating.rename(columns={"rating": "avg_rating"}, inplace=True)

        # Nombre d'interactions = 0
        count_zero = ratings_zero.groupby("item_id")["rating"].count().reset_index()
        count_zero.rename(columns={"rating": "num_ratings_zero"}, inplace=True)

        # Fusionner toutes les stats
        book_stats = books_df[["item_id", "title"]].merge(count_positive, on="item_id", how="left") \
                                                   .merge(avg_rating, on="item_id", how="left") \
                                                   .merge(count_zero, on="item_id", how="left")

        # Remplacer les NaN par 0
        book_stats[["num_ratings_positive", "avg_rating", "num_ratings_zero"]] = \
            book_stats[["num_ratings_positive", "avg_rating", "num_ratings_zero"]].fillna(0)

        # Trier par nombre de ratings positifs
        top_books = book_stats.sort_values("num_ratings_positive", ascending=False).head(10)

        # Afficher le tableau
        st.dataframe(top_books[["title", "num_ratings_positive", "avg_rating", "num_ratings_zero"]])

        # Barplot horizontal du nombre de ratings positifs
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="num_ratings_positive",
            y="title",
            data=top_books,
            palette="viridis",
            orient="h"
        )
        ax.set_xlabel("Nombre de ratings > 0")
        ax.set_ylabel("Titre du livre")
        st.pyplot(fig)

    else:
        st.info("Sélectionnez le dataset 'ratings' pour afficher les livres populaires.")

with tab3:
        # --- Tab : Auteurs les plus populaires ---
    if dataset_choice == "books":
        st.subheader("Top 10 auteurs avec le plus de livres dans le dataset")

        # Compter le nombre de livres par auteur
        author_counts = df["authors"].value_counts().reset_index()
        author_counts.columns = ["author", "num_books"]

        # Prendre le top 10
        top_authors = author_counts.head(10)

        # Afficher le tableau
        st.dataframe(top_authors)

        # Barplot horizontal
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            x="num_books",
            y="author",
            data=top_authors,
            palette="magma",
            orient="h"
        )
        ax.set_xlabel("Nombre de livres")
        ax.set_ylabel("Auteur")
        st.pyplot(fig)
    else:
        st.info("Sélectionnez le dataset 'books' pour afficher les auteurs les plus populaires.")

with tab4:
    if dataset_choice == "ratings":
        ratings_filtered = df[df["rating"] >= 0]  
        bins = st.slider("Nombre de classes pour l'histogramme", min_value=5, max_value=20, value=10)
        show_kde = st.checkbox("Afficher la courbe KDE", value=True)

        # Créer deux colonnes
        col1, col2 = st.columns(2)

        # --- Colonne 1 : Histogramme des notes ---
        with col1:
            st.subheader("Histogramme des notes")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(ratings_filtered["rating"], bins=bins, kde=show_kde, color="skyblue", ax=ax)
            ax.set_xlabel("Note")
            ax.set_ylabel("Nombre de ratings")
            st.pyplot(fig)

            counts, bin_edges = np.histogram(ratings_filtered["rating"], bins=bins)
            hist_df = pd.DataFrame({
                "Bin": [f"{round(bin_edges[i],2)} - {round(bin_edges[i+1],2)}" for i in range(len(bin_edges)-1)],
                "Nombre de ratings": counts
            })
            st.subheader("Nombre de ratings par bin")
            st.dataframe(hist_df)


        # --- Colonne 2 : Barplot proportion des types de notes ---
        with col2:
            st.subheader("Proportion des types de notes")
            counts = pd.Series({
                "0 (implicite)": (df["rating"] == 0).sum(),
                ">0 (réel)": (df["rating"] > 0).sum()
            })
            counts_df = counts.reset_index()
            counts_df.columns = ["Type de note", "Nombre"]
            counts_df["Proportion"] = counts_df["Nombre"] / counts_df["Nombre"].sum()

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x="Type de note",
                y="Proportion",
                data=counts_df,
                palette=["red", "green"],
                ax=ax
            )
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            st.subheader("Valeurs exactes")
            st.dataframe(counts_df)
    else:
        st.info("Sélectionnez le dataset 'ratings' pour afficher la distribution des notes.")


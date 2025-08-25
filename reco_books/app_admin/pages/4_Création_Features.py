import streamlit as st
import pandas as pd
import gensim
import os

from recommandation_de_livres.config import PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.iads.content_utils import combine_text
from recommandation_de_livres.iads.text_cleaning import nettoyage_leger, nettoyage_avance

# Réduire les logs TensorFlow parasites
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Génération de Features", layout="wide", page_icon="🧠")

# -------------------------------
# Dataset choisi depuis accueil
# -------------------------------
DIR = st.session_state.get("DIR", None)

if not DIR:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à l'accueil.")
    st.stop()

dataset_path = PROCESSED_DATA_DIR / DIR / "content_dataset.parquet"
if not dataset_path.exists():
    st.error(f"Fichier introuvable : {dataset_path}")
    st.stop()

st.title(f"🧠 Génération des Features - {DIR}")

tab1, tab2, tab3 = st.tabs(["Sentence-BERT", "Word2Vec", "TF-IDF"])

# =====================================================
# ======== TAB 1 : Sentence-BERT ======================
# =====================================================
with tab1:
    st.subheader("🔤 Features Sentence-BERT")
    st.info("Nettoie et fusionne les textes pour un futur encodage avec Sentence-BERT.")

    if st.button("Lancer la génération SBERT"):
        progress = st.progress(0, text="Initialisation...")

        try:
            # 1. Chargement
            progress.progress(10, text="Chargement du dataset...")
            content_df = load_parquet(dataset_path)

            # 2. Fusion des textes
            progress.progress(30, text="Fusion des champs texte...")
            content_df['text_combined'] = content_df.progress_apply(combine_text, axis=1)

            # 3. Nettoyage léger
            progress.progress(60, text="Nettoyage...")
            content_df['text_clean'] = content_df['text_combined'].progress_apply(nettoyage_leger)

            # 4. DataFrame features
            progress.progress(80, text="Création du DataFrame...")
            features_df = pd.DataFrame({
                'item_id': content_df['item_id'],
                'text_clean': content_df['text_clean']
            })

            # 5. Sauvegarde
            output_path_csv = PROCESSED_DATA_DIR / DIR / "features_sbert.csv"
            output_path_parquet = PROCESSED_DATA_DIR / DIR / "features_sbert.parquet"
            output_path_csv.parent.mkdir(parents=True, exist_ok=True)

            save_df_to_csv(features_df, output_path_csv)
            save_df_to_parquet(features_df, output_path_parquet)

            progress.progress(100, text="Terminé ✅")
            st.success("Features SBERT générées avec succès !")

            # Téléchargement direct
            st.download_button(
                label="Télécharger CSV",
                data=features_df.to_csv(index=False).encode("utf-8"),
                file_name="features_sbert.csv",
                mime="text/csv",
            )

            st.download_button(
                label="Télécharger Parquet",
                data=features_df.to_parquet(index=False),
                file_name="features_sbert.parquet",
                mime="application/octet-stream",
            )

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# =====================================================
# ======== TAB 2 : Word2Vec ===========================
# =====================================================
with tab2:
    st.subheader("🔤 Features Word2Vec")
    st.info("Nettoie, tokenise et prépare les textes pour un entraînement Word2Vec.")

    if st.button("Lancer la génération Word2Vec"):
        progress = st.progress(0, text="Initialisation...")

        try:
            # 1. Chargement
            progress.progress(10, text="Chargement du dataset...")
            content_df = load_parquet(dataset_path)

            # 2. Fusion des textes
            progress.progress(30, text="Fusion des champs texte...")
            content_df['text_combined'] = content_df.progress_apply(combine_text, axis=1)

            # 3. Nettoyage + tokenisation
            progress.progress(60, text="Tokenisation...")
            content_df['text_clean'] = content_df['text_combined'].progress_apply(gensim.utils.simple_preprocess)

            # 4. DataFrame features
            progress.progress(80, text="Création du DataFrame...")
            features_df = pd.DataFrame({
                'item_id': content_df['item_id'],
                'text_clean': content_df['text_clean']
            })

            # 5. Sauvegarde
            output_path_csv = PROCESSED_DATA_DIR / DIR / "features_w2v.csv"
            output_path_parquet = PROCESSED_DATA_DIR / DIR / "features_w2v.parquet"
            output_path_csv.parent.mkdir(parents=True, exist_ok=True)

            save_df_to_csv(features_df, output_path_csv)
            save_df_to_parquet(features_df, output_path_parquet)

            progress.progress(100, text="Terminé ✅")
            st.success("Features Word2Vec générées avec succès !")

            # Téléchargement direct
            st.download_button(
                label="Télécharger CSV",
                data=features_df.to_csv(index=False).encode("utf-8"),
                file_name="features_w2v.csv",
                mime="text/csv",
            )

            st.download_button(
                label="Télécharger Parquet",
                data=features_df.to_parquet(index=False),
                file_name="features_w2v.parquet",
                mime="application/octet-stream",
            )

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

with tab3:
    st.subheader("📊 Features TF-IDF")
    st.info("Prépare les textes (titres + auteurs) pour un entraînement TF-IDF.")

    if st.button("Lancer la génération TF-IDF"):
        progress = st.progress(0, text="Initialisation...")

        try:
            # 1. Chargement
            progress.progress(10, text="Chargement du dataset...")
            content_df = load_parquet(dataset_path)

            # 2. Préparation des textes (concaténation titre + auteurs)
            progress.progress(30, text="Préparation des champs texte...")
            content_df['text_for_tfidf'] = content_df.progress_apply(
                lambda row: f"{row['title']} {row['authors']}".strip(), axis=1
            )

            # 3. Nettoyage avancé
            progress.progress(60, text="Nettoyage avancé des textes...")
            content_df['text_clean'] = content_df['text_for_tfidf'].progress_apply(nettoyage_avance)

            # 4. DataFrame features
            progress.progress(80, text="Création du DataFrame...")
            features_df = pd.DataFrame({
                'item_id': content_df['item_id'],
                'text_clean': content_df['text_clean']
            })

            # 5. Sauvegarde
            output_path_csv = PROCESSED_DATA_DIR / DIR / "features_tfidf.csv"
            output_path_parquet = PROCESSED_DATA_DIR / DIR / "features_tfidf.parquet"
            output_path_csv.parent.mkdir(parents=True, exist_ok=True)

            save_df_to_csv(features_df, output_path_csv)
            save_df_to_parquet(features_df, output_path_parquet)

            progress.progress(100, text="Terminé ✅")
            st.success("Features TF-IDF générées avec succès !")

            # Téléchargement direct
            st.download_button(
                label="Télécharger CSV",
                data=features_df.to_csv(index=False).encode("utf-8"),
                file_name="features_tfidf.csv",
                mime="text/csv",
            )

            st.download_button(
                label="Télécharger Parquet",
                data=features_df.to_parquet(index=False),
                file_name="features_tfidf.parquet",
                mime="application/octet-stream",
            )

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

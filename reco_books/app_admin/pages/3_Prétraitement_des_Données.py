import streamlit as st
import pandas as pd
from pathlib import Path
from recommandation_de_livres.build_dataset import build_content_dataset, build_collaborative_dataset
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.config import RAW_DATA_DIR, INTERIM_DATA_DIR
import time

st.set_page_config(page_title="Prétraitement des données", layout="wide", page_icon="🛠️")


if 'DIR' not in st.session_state:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à la page d'accueil pour en choisir un.")
    st.stop()

# -------------------------------
# Dataset déjà choisi à l'accueil
# -------------------------------
DIR = st.session_state['DIR']
st.title(f"🛠️ Prétraitement contenu - {DIR}")

dataset_folder = RAW_DATA_DIR / DIR
interim_folder = INTERIM_DATA_DIR / DIR

if not dataset_folder.exists():
    st.error(f"Le dossier RAW pour le dataset '{DIR}' n'existe pas.")
    st.stop()
if not interim_folder.exists():
    interim_folder.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Lister les fichiers disponibles
# -------------------------------
file_dict = {}
for f in dataset_folder.glob("*.*"):
    file_dict[f.name] = dataset_folder
for f in interim_folder.glob("*.*"):
    file_dict[f.name] = interim_folder

file_names = list(file_dict.keys())

tab1, tab2 = st.tabs(['Basé sur le contenu', 'Collaboratif'])

# -------------------------------
# Choix des fichiers
# -------------------------------
with tab1:

    st.header("📊 Prétraitement Content-Based")

    books_file_name = st.selectbox("Fichier livres (books_uniform)", file_names, key="book_content")
    authors_file_name = st.selectbox("Fichier auteurs (optionnel)", ["None"] + file_names, key="authors_content")
    categories_file_name = st.selectbox("Fichier catégories (optionnel)", ["None"] + file_names, key="categories_content")

    books_file = file_dict[books_file_name] / books_file_name
    authors_file = file_dict[authors_file_name] / authors_file_name if authors_file_name != "None" else None
    categories_file = file_dict[categories_file_name] / categories_file_name if categories_file_name != "None" else None

    # -------------------------------
    # Préchargement pour récupérer les colonnes
    # -------------------------------
    if books_file.suffix == ".csv":
        books_df_preview = pd.read_csv(books_file, nrows=5)
    else:
        books_df_preview = pd.read_parquet(books_file, engine="pyarrow")

    columns = list(books_df_preview.columns)

    # -------------------------------
    # Choix des colonnes
    # -------------------------------
    st.subheader("Mapping des colonnes")
    title_col = st.selectbox("Colonne titre", columns, index=columns.index("title") if "title" in columns else 0)
    desc_col = st.selectbox("Colonne description", ["None"] + columns, index=columns.index("description") if "description" in columns else 0)
    authors_col = st.selectbox("Colonne auteurs", columns, index=columns.index("authors") if "authors" in columns else 0)
    book_id_col = st.selectbox("Colonne ID livre", columns, index=columns.index("item_id") if "item_id" in columns else 0)
    lang_col = st.selectbox("Colonne langue", ["None"] + columns, index=columns.index("language") if "language" in columns else 0)

    # -------------------------------
    # Paramètres supplémentaires
    # -------------------------------
    add_lang = st.checkbox("Ajouter les langues manquantes", value=False)
    allowed_langs = st.text_input("Langues autorisées (séparées par virgule)", "en,eng,en-US,en-GB,en-CA")
    allowed_langs = [lang.strip() for lang in allowed_langs.split(",")]

    get_description = st.checkbox("Ajouter les descriptions manquantes (Uniquement si manquantes)", value=False)

    st.caption(
    "⚠️ Prérequis : Une base de données PostgreSQL créée depuis la base de données d'openlibrary : https://openlibrary.org/developers/dumps\n"
    "Lien pour créer la base : https://github.com/LibrariesHacked/openlibrary-search\n"
    "Ce paramètre ne concerne que les datasets qui n'ont pas de colonne 'description'."
    )


    # -------------------------------
    # Bouton pour lancer le prétraitement
    # -------------------------------
    if st.button("✅ Lancer le prétraitement"):

        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = [
            "Chargement des livres",
            "Chargement des auteurs",
            "Chargement des catégories",
            "Build content dataset",
            "Prévisualisation",
            "Sauvegarde"
        ]
        total_steps = len(steps)

        # --- Étape 1 : charger books ---
        status_text.text(steps[0])
        if books_file.suffix == ".csv":
            books_df = pd.read_csv(books_file)
        else:
            books_df = pd.read_parquet(books_file)
        progress_bar.progress(1 / total_steps)
        time.sleep(0.2)

        # --- Étape 2 : charger auteurs ---
        if authors_file:
            status_text.text(steps[1])
            if authors_file.suffix == ".csv":
                authors_df = pd.read_csv(authors_file)
            else:
                authors_df = pd.read_parquet(authors_file)
        else:
            authors_df = None
        progress_bar.progress(2 / total_steps)
        time.sleep(0.2)

        # --- Étape 3 : charger catégories ---
        if categories_file:
            status_text.text(steps[2])
            if categories_file.suffix == ".csv":
                categories_df = pd.read_csv(categories_file)
            else:
                categories_df = pd.read_parquet(categories_file)
        else:
            categories_df = None
        progress_bar.progress(3 / total_steps)
        time.sleep(0.2)

        # --- Étape 4 : build content dataset ---
        status_text.text(steps[3])
        books_df = build_content_dataset.build_content_dataset(
            books=books_df,
            authors=authors_df,
            categories=categories_df,
            dataset_dir=interim_folder,
            title_col=title_col,
            desc_col=None if desc_col=="None" else desc_col,
            authors_col=authors_col,
            book_id_col=book_id_col,
            lang_col=None if lang_col=="None" else lang_col,
            add_language=add_lang,
            get_description=get_description,
            allowed_langs=allowed_langs
        )
        progress_bar.progress(4 / total_steps)
        time.sleep(0.2)

        # --- Étape 5 : prévisualisation ---
        status_text.text(steps[4])
        st.subheader("Prévisualisation du dataset final")
        st.write(f"Dimensions : {books_df.shape}")
        st.dataframe(books_df.head(20))
        st.write("Types de colonnes :")
        st.dataframe(pd.DataFrame(books_df.dtypes, columns=["Type"]))
        st.write("Pourcentage de valeurs manquantes par colonne :")
        st.dataframe((books_df.isnull().mean() * 100).round(2))
        progress_bar.progress(5 / total_steps)
        time.sleep(0.2)

        # --- Étape 6 : sauvegarde ---
        status_text.text(steps[5])
        save_df_to_csv(books_df, interim_folder / "books_content_dataset.csv")
        save_df_to_parquet(books_df, interim_folder / "books_content_dataset.parquet")
        progress_bar.progress(1.0)

        status_text.text("✅ Prétraitement terminé !")
        st.success(f"Dataset sauvegardé dans : {interim_folder}")

        # --- Après la sauvegarde ---
        status_text.text("✅ Prétraitement terminé !")
        st.success(f"Dataset sauvegardé dans : {interim_folder}")

        # --- Boutons de téléchargement ---
        st.subheader("Télécharger le dataset prétraité")
        csv_file_path = interim_folder / "books_content_dataset.csv"
        parquet_file_path = interim_folder / "books_content_dataset.parquet"

        # CSV
        with open(csv_file_path, "rb") as f:
            st.download_button(
                label="📥 Télécharger CSV",
                data=f,
                file_name="books_content_dataset.csv",
                mime="text/csv"
            )

        # Parquet
        with open(parquet_file_path, "rb") as f:
            st.download_button(
                label="📥 Télécharger Parquet",
                data=f,
                file_name="books_content_dataset.parquet",
                mime="application/octet-stream"
            )

with tab2:  # Tab Collaborative
    st.header("📊 Prétraitement Collaborative Filtering")

    # Fichiers nécessaires
    books_file_name = st.selectbox("Fichier livres (books_uniform)", file_names, key="book_collab")
    ratings_file_name = st.selectbox("Fichier ratings (ratings_uniform)", file_names, key="ratings_collab")
    content_file_name = st.selectbox("Fichier content_dataset (optionnel)", file_names + ["None"], key="content_collab")
    
    books_file = file_dict[books_file_name] / books_file_name
    ratings_file = file_dict[ratings_file_name] / ratings_file_name
    content_file = file_dict[content_file_name] / content_file_name if content_file_name != "None" else None

    min_ratings = st.number_input("Minimum de ratings par livre", min_value=0, value=0)
    min_users_interaction = st.number_input("Minimum d'interactions par utilisateur", min_value=0, value=100)

    if st.button("✅ Lancer le prétraitement collaborative"):

        progress_bar = st.progress(0)
        status_text = st.empty()
        steps = [
            "Chargement des livres",
            "Chargement des ratings",
            "Chargement du content dataset",
            "Construction du collaborative dataset",
            "Ajout index utilisateurs",
            "Prévisualisation",
            "Sauvegarde des fichiers"
        ]
        total_steps = len(steps)

        # --- Chargement des fichiers ---
        status_text.text(steps[0])
        if books_file.suffix == ".csv":
            books_df = pd.read_csv(books_file)
        else:
            books_df = pd.read_parquet(books_file)
        progress_bar.progress(1 / total_steps)

        status_text.text(steps[1])
        if ratings_file.suffix == ".csv":
            ratings_df = pd.read_csv(ratings_file)
        else:
            ratings_df = pd.read_parquet(ratings_file)
        progress_bar.progress(2 / total_steps)

        status_text.text(steps[2])
        if content_file:
            if content_file.suffix == ".csv":
                content_df = pd.read_csv(content_file)
            else:
                content_df = pd.read_parquet(content_file)
        else:
            content_df = None
        progress_bar.progress(3 / total_steps)

        # --- Construction du collaborative dataset ---
        status_text.text(steps[3])
        collab_df = build_collaborative_dataset.build_collaborative_dataset(
            books=content_df if content_df is not None else books_df,
            ratings=ratings_df,
            authors=None,
            min_ratings=min_ratings,
            min_users_interaction=min_users_interaction
        )
        progress_bar.progress(4 / total_steps)

        # --- Prévisualisation ---
        status_text.text(steps[5])
        st.subheader("Aperçu du dataset collaboratif")
        st.write(f"Dimensions : {collab_df.shape}")
        st.dataframe(collab_df.head(20))
        progress_bar.progress(6 / total_steps)

        # --- Sauvegarde ---
        status_text.text(steps[6])
        collab_csv_path = INTERIM_DATA_DIR / DIR / "collaborative_dataset.csv"
        collab_parquet_path = INTERIM_DATA_DIR / DIR / "collaborative_dataset.parquet"
        users_path = INTERIM_DATA_DIR / DIR / "users.csv"

        save_df_to_csv(collab_df, collab_csv_path)
        save_df_to_parquet(collab_df, collab_parquet_path)

        # Création fichier utilisateurs
        from recommandation_de_livres.iads.create_users import create_users_file
        create_users_file(collab_csv_path, users_path)

        progress_bar.progress(1.0)
        status_text.text("✅ Prétraitement collaboratif terminé !")
        st.success(f"Fichiers sauvegardés dans {INTERIM_DATA_DIR / DIR}")

        # --- Boutons téléchargement ---
        with open(collab_csv_path, "rb") as f:
            st.download_button("📥 Télécharger CSV", f, file_name="collaborative_dataset.csv", mime="text/csv")
        with open(collab_parquet_path, "rb") as f:
            st.download_button("📥 Télécharger Parquet", f, file_name="collaborative_dataset.parquet", mime="application/octet-stream")
        with open(users_path, "rb") as f:
            st.download_button("📥 Télécharger fichier utilisateurs", f, file_name="users.csv", mime="text/csv")

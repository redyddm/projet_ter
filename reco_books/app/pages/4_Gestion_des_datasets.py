import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="📂 Gestion des Datasets", layout="wide")
st.title("📂 Import et uniformisation des datasets")

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

tab1, tab2 = st.tabs(["Ratings", "Books"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV, Parquet ou Excel", type=["csv", "parquet", "xlsx"], key="ratings_upload"
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Aperçu du dataset")
        st.dataframe(df.head())

        st.subheader("🛠️ Mapping des colonnes pour uniformisation")
        user_col = st.selectbox("Colonne utilisateur (user_id)", ["None"] + list(df.columns), key="user_col")
        item_col = st.selectbox("Colonne item/livre (item_id)", ["None"] + list(df.columns), key="item_col")
        rating_col = st.selectbox("Colonne note (rating)", ["None"] + list(df.columns), key="rating_col")

        if st.button("✅ Valider et uniformiser Ratings", key="validate_ratings"):
            standardized_df = pd.DataFrame()

            for col, name in zip([user_col, item_col, rating_col], ["user_id", "item_id", "rating"]):
                if col != "None":
                    standardized_df[name] = df[col]
                else:
                    st.warning(f"La colonne {name} n'a pas été sélectionnée. Vérifiez vos mappings.")

            standardized_df = standardized_df.dropna(subset=["user_id", "item_id", "rating"])
            standardized_df["rating"] = pd.to_numeric(standardized_df["rating"], errors="coerce")
            standardized_df = standardized_df.dropna(subset=["rating"])

            st.success("Dataset uniformisé avec succès !")
            st.dataframe(standardized_df.head())

            save_path = DATA_DIR / "ratings_uniform.csv"
            standardized_df.to_csv(save_path, index=False)
            st.info(f"Dataset sauvegardé sous : `{save_path}`")

with tab2:
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV, Parquet ou Excel", type=["csv", "parquet", "xlsx"], key="books_upload"
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Aperçu du dataset")
        st.dataframe(df.head())

        st.subheader("🛠️ Mapping des colonnes pour uniformisation")
        item_col = st.selectbox("Colonne item/livre (item_id)", ["None"] + list(df.columns), key="book_item_col")
        title_col = st.selectbox("Titre du livre", ["None"] + list(df.columns), key="book_title_col")
        author_col = st.selectbox("Auteur", ["None"] + list(df.columns), key="book_author_col")
        publisher_col = st.selectbox("Éditeur", ["None"] + list(df.columns), key="book_publisher_col")
        year_col = st.selectbox("Année de publication", ["None"] + list(df.columns), key="book_year_col")
        image_col = st.selectbox("Image URL", ["None"] + list(df.columns), key="book_image_col")

        if st.button("✅ Valider et uniformiser Books", key="validate_books"):
            standardized_df = pd.DataFrame()

            # Obligatoire
            if item_col != "None":
                standardized_df["item_id"] = df[item_col]
            else:
                st.warning("La colonne item_id n'a pas été sélectionnée. Vérifiez vos mappings.")

            # Optionnelles
            for col, name in zip([title_col, author_col, publisher_col, year_col, image_col],
                                 ["title", "authors", "publisher", "year", "image_url"]):
                if col != "None":
                    standardized_df[name] = df[col]

            st.success("Dataset uniformisé avec succès !")
            st.dataframe(standardized_df.head())

            save_path = DATA_DIR / "books_uniform.csv"
            standardized_df.to_csv(save_path, index=False)
            st.info(f"Dataset sauvegardé sous : `{save_path}`")

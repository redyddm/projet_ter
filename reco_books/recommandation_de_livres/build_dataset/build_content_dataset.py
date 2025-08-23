from recommandation_de_livres.preprocessing.preprocess_content import *
from recommandation_de_livres.config import INTERIM_DATA_DIR 
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.iads.text_cleaning import clean_text_for_display
from pathlib import Path
import numpy as np

def build_content_dataset(books, authors=None, categories=None, dataset_dir=None,
                          title_col="title", desc_col=None,
                          authors_col="authors", book_id_col="item_id",
                          lang_col="language", add_language=False, allowed_langs=None):
    """
    Build content dataset uniformisé avec :
    - mapping des auteurs
    - ajout des langues manquantes
    - ajout des catégories/genres
    - filtrage et nettoyage (NaN, doublons, langues)
    """

    # --- Mapping des auteurs si fourni ---
    if authors is not None:
        books = map_author_names(books, authors, authors_col=authors_col)

    # --- Sauvegarde intermédiaire après mapping auteurs ---
    save_df_to_csv(books, dataset_dir / "books_authors.csv")
    save_df_to_parquet(books, dataset_dir / "books_authors.parquet")

    # --- Ajout des catégories si fourni ---
    if categories is not None:
        books = add_genres_str_column(books, categories, book_id_col="book_id")

    if desc_col is None:
        books = get_infos(books)
        desc_col="description"
        catego_col="categories"

    # --- Nettoyage des balises html et du bruit ---

    books['title'] = books['title'].apply(clean_text_for_display)
    books['authors'] = books['authors'].apply(clean_text_for_display)
    books['categories'] = books['categories'].apply(clean_text_for_display)
    books['publisher'] = books['publisher'].apply(clean_text_for_display)
    books['description'] = books['description'].apply(clean_text_for_display)

    # --- Ajout de la langue pour les lignes où elle est manquante ---
    if add_language:
        books = add_language_column(books, title_col=title_col, desc_col=desc_col, lang_col=lang_col)

    # --- Filtrage de base (NaN sur titre/description et langues autorisées) ---
    books = filter_books_basic(books, title_col=title_col, desc_col=desc_col,
                               lang_col=lang_col, allowed_langs=allowed_langs)

    # --- Suppression des doublons sur titre nettoyé ---
    books = remove_duplicates(books, title_col=title_col)

    #

    return books

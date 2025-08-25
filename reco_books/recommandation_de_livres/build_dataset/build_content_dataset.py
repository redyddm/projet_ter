from recommandation_de_livres.preprocessing.preprocess_content import *
from recommandation_de_livres.config import INTERIM_DATA_DIR 
from recommandation_de_livres.iads.utils import save_df_to_csv, save_df_to_parquet
from recommandation_de_livres.iads.text_cleaning import nettoyage_balises
from pathlib import Path
import numpy as np

def build_content_dataset(books, authors=None, categories=None, dataset_dir=None,
                          title_col="title", desc_col=None,
                          authors_col="authors", book_id_col="item_id",
                          lang_col="language", add_language=False, get_description=False, 
                          allowed_langs=None, update_db=False):
    """Construit un dataset de contenu nettoyé et enrichi à partir d'un DataFrame de livres.
    
    Étapes réalisées :
    - Mapping des auteurs via `authors` si fourni.
    - Ajout des catégories/genres via `categories` si fourni.
    - Récupération des descriptions manquantes si `get_description=True`.
    - Nettoyage des balises HTML et du bruit textuel.
    - Ajout de la langue manquante si `add_language=True`.
    - Filtrage (NaN, langues autorisées).
    - Suppression des doublons sur le titre nettoyé.

    Args:
        books (pd.DataFrame) : DataFrame contenant les informations des livres.
        authors (pd.DataFrame, optionnel) : DataFrame des auteurs avec `author_id` et `name` pour mapping.
        categories (pd.DataFrame, optionnel) : DataFrame des catégories/genres associés aux livres.
        dataset_dir (Path, optionnel) : Répertoire de sauvegarde intermédiaire du dataset.
        title_col (str) : Nom de la colonne contenant les titres des livres.
        desc_col (str, optionnel) : Colonne des descriptions. Si None et `get_description=True`, sera récupérée.
        authors_col (str) : Colonne contenant les IDs des auteurs (par défaut "authors").
        book_id_col (str) : Colonne identifiant unique des livres (par défaut "item_id").
        lang_col (str) : Colonne contenant le code de langue.
        add_language (bool) : Si True, détecte et ajoute la langue manquante.
        get_description (bool) : Si True, tente de récupérer les descriptions manquantes.
        allowed_langs (list, optionnel) : Liste des codes langues autorisées (ex. ['en', 'eng']).
        update_db (bool) : Si True, met à jour la base de données lors de la récupération des descriptions.
    
    Returns:
        pd.DataFrame : DataFrame nettoyé, enrichi et prêt pour un système de recommandation basé sur le contenu.
    """

    # --- Mapping des auteurs si fourni ---
    if authors is not None:
        books = map_author_names(books, authors, authors_col=authors_col)

    # --- Ajout des catégories si fourni ---
    if categories is not None:
        books = add_genres_str_column(books, categories, book_id_col=book_id_col)

    # --- Récupération des descriptions si demandé ---
    if desc_col is None and get_description:
        books = get_infos(books, update=update_db)
        desc_col = "description"
        catego_col = "categories"

    # --- Nettoyage du texte ---
    books['title'] = books['title'].apply(nettoyage_balises)
    books['authors'] = books['authors'].apply(nettoyage_balises)
    books['categories'] = books['categories'].apply(nettoyage_balises)
    books['publisher'] = books['publisher'].apply(nettoyage_balises)
    books['description'] = books['description'].apply(nettoyage_balises)

    # --- Ajout de la langue manquante ---
    if add_language:
        books = add_language_column(books, title_col=title_col, desc_col=desc_col, lang_col=lang_col)

    # --- Filtrage de base ---
    books = filter_books_basic(books, title_col=title_col, desc_col=desc_col,
                               lang_col=lang_col, allowed_langs=allowed_langs)

    # --- Suppression des doublons ---
    books = remove_duplicates(books, title_col=title_col)

    return books

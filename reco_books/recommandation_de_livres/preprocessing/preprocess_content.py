import ast
from tqdm import tqdm
import pandas as pd
from langdetect import detect, DetectorFactory
from recommandation_de_livres.iads.text_cleaning import nettoyage_avance
from open_library.openlibrary_extract import *

DetectorFactory.seed = 0

def str_to_list(s):
    return ast.literal_eval(s)

def list_to_str(lst):
    if isinstance(lst, list):
        return ' '.join(str(x) for x in lst)
    return str(lst)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def filter_books_basic(books, title_col="title", desc_col="description", lang_col=None, allowed_langs=None):
    """
    Filtre les livres vides ou non anglais.
    allowed_langs: liste de codes de langue (ex: ['en', 'eng', 'en-US'])
    """
    books = books.copy()
    # Filtre sur titre et description
    books = books.dropna(subset=[title_col, desc_col])
    
    # Filtre sur langue si la colonne est fournie
    if lang_col and allowed_langs:
        books = books[books[lang_col].isin(allowed_langs) | books[lang_col].isnull()].copy()
    
    return books

def remove_duplicates(books, title_col="title"):
    """
    Supprime les doublons par titre nettoyé.
    """
    tqdm.pandas(desc="Nettoyage des textes")
    books['title_clean'] = books[title_col].progress_apply(nettoyage_avance)
    books = books.drop_duplicates(subset='title_clean').reset_index(drop=True)
    return books

def add_language_column(books, title_col="title", desc_col="description", lang_col="language"):
    """
    Ajoute une colonne de langue uniquement pour les lignes où elle est NaN.
    """
    books = books.copy()
    if lang_col not in books.columns:
        books[lang_col] = None
    
    tqdm.pandas(desc="Détection de la langue")
    mask = books[lang_col].isna()
    books.loc[mask, lang_col] = (
        (books.loc[mask, title_col].fillna('') + ' ' + books.loc[mask, desc_col].fillna(''))
        .progress_apply(detect_language)
    )
    return books

def get_descriptions(books): 

    isbn_list = books['isbn'] 
    
    keys=get_key_books_list(isbn_list) 

    update_editions_work_key(keys) 

    descriptions, isbn_13 = get_used_infos_by_isbn_list(isbn_list) 

    books['description'] = descriptions 

    books['isbn13'] = isbn_13 

    return books

def map_author_names(books_df, authors_df, authors_col="authors"):
    """
    Remplace les author_id dans books_df['authors'] par les noms depuis authors_df['name'].
    """
    books_df = books_df.copy()
    author_map = dict(zip(authors_df['author_id'].astype(str), authors_df['name']))

    def extract_names(author_list):
    # Si c'est une chaîne → essayer de parser
        if isinstance(author_list, str):
            try:
                author_list = ast.literal_eval(author_list)
            except Exception:
                return ""
        if not isinstance(author_list, list):
            return ""
        
        names = []
        for a in author_list:
            author_id = str(a.get("author_id"))
            name = author_map.get(author_id)
            if pd.notna(name):
                names.append(str(name))
        return ", ".join(names)

    tqdm.pandas(desc="Mapping des auteurs")
    books_df[authors_col] = books_df[authors_col].progress_apply(extract_names)
    return books_df

def add_categories_columns(books, categories_df, book_id_col="book_id"):
    """
    Merge avec les catégories/genres si disponibles.
    """
    books = books.merge(categories_df, on=book_id_col, how='left')
    return books
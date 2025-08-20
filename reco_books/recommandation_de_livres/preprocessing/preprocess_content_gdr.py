import ast
from tqdm import tqdm
import gensim
import numpy as np
import pandas as pd

from open_library.openlibrary_extract import *
from recommandation_de_livres.iads.text_cleaning import nettoyage_texte, nettoyage_avance

from langdetect import detect, DetectorFactory

def str_to_list(s):
    return ast.literal_eval(s)

def list_to_str(lst):
    if isinstance(lst, list):
        return ' '.join(str(x) for x in lst)
    return str(lst)

def filter_books_basic(books):
    """Filtre les livres vides ou non anglais."""

    books_df = books[books['language_code'].isin(['en', 'eng','en-US','en-GB','en-CA'])].copy()
    books_df = books_df.dropna(subset=['description', 'title'])

    return books_df

def remove_duplicates(books):
    
    tqdm.pandas(desc="Nettoyage des textes")
    books['title_clean']=books['title'].progress_apply(nettoyage_avance)
    books = books.drop_duplicates(subset='title_clean')
    books = books.reset_index(drop=True)

    return books

def map_author_names(books_df, authors_df):
    """
    Remplace les author_id dans books_df['authors'] par les noms depuis authors_df['name'].
    """

    # Dictionnaire author_id -> name
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
    books_df["authors"] = books_df["authors"].progress_apply(extract_names)

    return books_df

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def add_language_column(books):
    """
    Ajoute une colonne 'language_code' uniquement pour les lignes où elle est NaN,
    en détectant la langue à partir du titre + description.
    """
    tqdm.pandas(desc="Detecting language")
    
    mask = books['language_code'].isna()
    books.loc[mask, 'language_code'] = (
        (books.loc[mask, 'title'].fillna('') + ' ' + books.loc[mask, 'description'].fillna(''))
        .progress_apply(detect_language)
    )
    
    return books

def add_categories_columns(books, categories):
    books_with_genres = books.merge(categories, on='book_id', how='left')
    return books_with_genres

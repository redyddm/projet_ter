import ast
from tqdm import tqdm
import gensim
import numpy as np

from open_library.openlibrary_extract import *
from recommandation_de_livres.iads.text_cleaning import nettoyage_texte, nettoyage_avance

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def str_to_list(s):
    return ast.literal_eval(s)

def list_to_str(lst):
    if isinstance(lst, list):
        return ' '.join(str(x) for x in lst)
    return str(lst)

def select_and_rename_books_columns(books):
    """
    Sélectionne les colonnes importantes et renomme isbn10 en isbn.
    """
    books.rename(columns={'ISBN':'isbn','Book-Title':'title', 'Book-Author': 'authors','Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)
    books_df = books.drop(columns=['Image-URL-S', 'Image-URL-M'])

    return books_df

def filter_books_basic(books):
    """Filtre les livres vides ou non anglais et supprime les doublons."""

    books_df = books[books['language']=='en'].copy()
    books_df = books_df[books_df['description'].apply(lambda x: len(x) > 0)]
    books_df.dropna(inplace=True)

    return books_df

def remove_duplicates(books):
    
    tqdm.pandas(desc="Nettoyage des textes")
    books['title_clean']=books['title'].progress_apply(nettoyage_avance)
    books = books.drop_duplicates(subset='title_clean')
    books = books.reset_index(drop=True)

    return books

def get_descriptions(books):

    isbn_list = books['isbn']


    keys=get_key_books_list(isbn_list)
    update_editions_work_key(keys)

    descriptions, isbn_13 = get_used_infos_by_isbn_list(isbn_list)

    books['description'] = descriptions
    books['isbn13'] = isbn_13

    return books

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def add_language_column(books):
    """
    Ajoute une colonne 'language' en détectant la langue à partir du titre + description
    """
    tqdm.pandas(desc="Detecting language")
    books['language'] = (books['title'].fillna('')).progress_apply(detect_language)
    return books

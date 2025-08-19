import ast
from tqdm import tqdm
import gensim
import numpy as np

from open_library.openlibrary_extract import *
from recommandation_de_livres.iads.text_cleaning import nettoyage_texte

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
    books = books[books['description'].apply(lambda x: len(x) > 0)]
    books.dropna(inplace=True)

    return books

def remove_duplicates(books):
    
    tqdm.pandas(desc="Nettoyage des textes")
    books['title_clean']=books['title'].progress_apply(nettoyage_texte)
    books = books.drop_duplicates(subset='title_clean')
    books = books.reset_index(drop=True)

    return books

def get_descriptions(books, update=True):

    isbn_list = books['isbn']

    if update: # Nécessaire que la première fois
        keys=get_key_books_list(isbn_list)
        update_editions_work_key(keys)

    descriptions, isbn_13 = get_used_infos_by_isbn_list(isbn_list)

    books['description'] = descriptions
    books['isbn13'] = isbn_13

    return books


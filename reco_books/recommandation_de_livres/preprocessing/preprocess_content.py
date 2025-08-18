import ast
from tqdm import tqdm
import gensim
import numpy as np

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
    books_df = books[['isbn10', 'isbn13', 'title', 'authors', 'categories', 'description', 
                      'lang', 'url', 'image-url', 'format', 'rating-avg']].copy()
    books_df.rename(columns={'isbn10': 'isbn'}, inplace=True)
    return books_df

def filter_books_basic(books):
    """Filtre les livres vides ou non anglais et supprime les doublons."""
    books = books[(books['authors'] != '[]') & (books['categories'] != '[]')]
    books = books[books['description'].notna() & (books['description'] != '[]')]
    books = books[books['lang'] == 'en']

    return books

def remove_duplicates(books):
    
    tqdm.pandas(desc="Nettoyage des textes")
    books['title_clean']=books['title'].progress_apply(nettoyage_texte)
    books = books.drop_duplicates(subset='title_clean')
    books = books.drop(columns='title_clean')
    books = books.reset_index(drop=True)

    return books

def map_ids_to_names(books, authors, categories):
    """Mappe les IDs authors/categories vers leurs noms."""
    
    books['authors'] = books['authors'].apply(str_to_list)
    books['categories'] = books['categories'].apply(str_to_list)

    author_map = dict(zip(authors['author_id'], authors['author_name']))
    category_map = dict(zip(categories['category_id'], categories['category_name']))

    tqdm.pandas(desc="Récupération des noms des auteurs")
    books['authors'] = books['authors'].progress_apply(lambda ids: [author_map[i] for i in ids])

    tqdm.pandas(desc="Récupération des noms des catégories")
    books['categories'] = books['categories'].progress_apply(lambda ids: [category_map[i] for i in ids])

    # Convertir en string
    books['authors'] = books['authors'].apply(list_to_str)
    books['categories'] = books['categories'].apply(list_to_str)

    # Filtrage des NaN restants 
    books = books[books['authors'].notna() & books['authors'].apply(lambda x: len(x) > 0)]
    books = books[books['categories'].notna() & books['categories'].apply(lambda x: len(x) > 0)]

    books = books.reset_index(drop=True)
    return books

def add_clean_title(books):
    books['title_clean'] = books['title'].apply(nettoyage_texte)
    return books

import ast
from tqdm import tqdm
import gensim
import numpy as np

from recommandation_de_livres.iads.text_cleaning import nettoyage_texte, nettoyage_avance


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
    books['title_clean']=books['title'].progress_apply(nettoyage_avance)
    books = books.drop_duplicates(subset='title_clean')
    books = books.drop(columns='title_clean')
    books = books.reset_index(drop=True)

    return books

def map_ids_to_names(books, authors, categories):
    """Mappe les IDs authors/categories vers leurs noms (rapide et robuste)."""
    
    # Convertir les colonnes authors/categories en listes (si ce n'est pas déjà fait)
    books['authors'] = books['authors'].apply(str_to_list)
    books['categories'] = books['categories'].apply(str_to_list)

    # Création des dictionnaires ID → nom
    author_map = dict(zip(authors['author_id'], authors['name']))
    category_map = dict(zip(categories['category_id'], categories['category_name']))

    tqdm.pandas(desc="Mapping auteurs")
    books['authors'] = books['authors'].progress_apply(
        lambda ids: ", ".join([author_map.get(i) for i in ids if i in author_map])
    )

    tqdm.pandas(desc="Mapping catégories")
    books['categories'] = books['categories'].progress_apply(
        lambda ids: ", ".join([category_map.get(i) for i in ids if i in category_map])
    )

    # Filtrage des lignes vides (sans auteurs ni catégories)
    books = books[(books['authors'].str.len() > 0) & (books['categories'].str.len() > 0)]

    # Réindexation propre
    return books.reset_index(drop=True)

def add_clean_title(books):
    books['title_clean'] = books['title'].apply(nettoyage_avance)
    return books

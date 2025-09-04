import ast
from tqdm import tqdm
import pandas as pd
from recommandation_de_livres.iads.utils import detect_lang
from recommandation_de_livres.iads.text_cleaning import nettoyage_avance, nettoyage_titre
from open_library.openlibrary_extract import *
from lingua import Language, LanguageDetectorBuilder

def list_to_str_desc(lst):
    """ Transforme une liste contenant une phrase en un str.
        Args:
            lst (str[]) : liste contenant la phrase
    """
    if isinstance(lst, list):
        return ' '.join(str(x) for x in lst)
    return str(lst)

def list_to_str_cate(lst):
    """ Transforme une liste de mots en un str avec des mots séparés par une virgule.
        Args:
            lst (str[]) : liste de mots 
    """
    if isinstance(lst, list):
        return ', '.join(str(x) for x in lst)
    return str(lst)

def get_infos(books, update=True): 
    """ Fonction qui permete d'ajouter les colonnes descriptions et categories
        avec les informations extraites de la base de données.
        Args:
            books (pd.DataFrame) : DataFrame de livres
            update (Bool) : Booléen qui sert à déclencher la mise à jour de la base de données (nécessaire la première fois)
        Returns:
            books_df (pd.DataFrame) : DataFrame mis à jour
    """
    books_df = books.copy()
    isbn_list = books_df['isbn'] # Récupération des isbn
    if update :
        keys=get_key_books_list(isbn_list) # Récupération des clés
        update_editions_work_key(keys) # Mise à jour de la base de données
    categories, descriptions = get_infos_by_isbn_list(isbn_list) # Récupération des informations

    # Ajout des informations dans des colonnes
    books_df['description'] = descriptions 
    books_df['categories'] = categories 

    # Traitement des textes obtenus pour les avoir en str
    books_df['description'] = books_df['description'].apply(list_to_str_desc)
    books_df['categories'] = books_df['categories'].apply(list_to_str_cate)

    return books_df

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

def detect_lang(text):
    """ Détecte la langue du texte donnée
        Args:
            text (str) : texte servant à déterminer la langue
    """
    # On gère une exception au cas où
    try:
        lang=detector.detect_language_of(text)
        return lang.iso_code_639_3.name.lower()
    
    except Exception as e:
        return ""
    
def add_language_column(books, title_col="title", desc_col="description", lang_col="language"):
    """ Ajoute une colonne de langue uniquement pour les lignes où elle est NaN.
        Args:
            books (pd.DataFrame) :  DataFrame de livres
            title_col (str) : Colonne des titres
            desc_col (str) : Colonne des descriptions
            lang_col (str) : Colonne des langues
        Returns:
            books (pd.DataFrame) : DataFrame des livres avec les langues ajoutées
    """
    books = books.copy()
    if lang_col not in books.columns:
        books[lang_col] = None
    
    tqdm.pandas(desc="Détection de la langue")
    mask = books[lang_col].isna()

    books.loc[mask, lang_col] = (
        books.loc[mask, title_col].fillna('')
        .progress_apply(detect_lang)
    )
    
    # Si on veut utiliser les descriptions

    #books.loc[mask, lang_col] = (
    #    (books.loc[mask, title_col].fillna('') + ' ' + books.loc[mask, desc_col].fillna(''))
    #    .progress_apply(detect_lang)
    #)
    return books

def filter_books_basic(books, title_col="title", desc_col="description", lang_col=None, allowed_langs=None):
    """ Filtre les livres vides ou non anglais.
        Args:
            books (pd.DataFrame) :  DataFrame de livres
            title_col (str) : Colonne des titres
            desc_col (str) : Colonne des descriptions
            lang_col (str) : Colonne des langues
            allowed_langs: liste de codes de langue (ex: ['en', 'eng', 'en-US'])
    """
    books = books.copy()
    
    # Supprime les titres NaN ou vides
    books = books[books[title_col].notna() & (books[title_col].str.strip() != "")]
    
    # Supprime les descriptions NaN ou vides
    books = books[books[desc_col].notna() & (books[desc_col].str.strip() != "")]
    
    # Filtre sur langue si la colonne est fournie
    if lang_col and allowed_langs:
        books = books[books[lang_col].isin(allowed_langs) | books[lang_col].isnull()].copy()
    
    return books

def remove_duplicates(books, title_col="title"):
    """ Supprime les doublons par titre nettoyé.
        Args:
            books (pd.DataFrame) :  DataFrame de livres
            title_col (str) : Colonne des titres
        Returns:
            books (pd.DataFrame) : DataFrame des livres sans doublon
    """
    tqdm.pandas(desc="Nettoyage des textes")
    books['title_clean'] = books[title_col].progress_apply(nettoyage_avance)
    books = books.drop_duplicates(subset='title_clean').reset_index(drop=True)
    books['title_clean'] = books[title_col].progress_apply(nettoyage_titre)
    
    return books

def map_author_names(books_df, authors_df, authors_col="authors"):
    """Remplace les author_id dans books_df[authors_col] par leurs noms depuis authors_df['name'].
    Args:
        books_df (pd.DataFrame) : DataFrame contenant les livres et leurs auteurs sous forme d'IDs.
        authors_df (pd.DataFrame) : DataFrame contenant les informations des auteurs (colonnes 'author_id' et 'name').
        authors_col (str) : Nom de la colonne dans books_df contenant les IDs des auteurs (par défaut 'authors').
    Returns:
        pd.DataFrame : DataFrame avec la colonne des auteurs remplacée par leurs noms (séparés par des virgules).
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
    """Ajoute les catégories/genres aux livres via un merge avec categories_df.
    Args:
        books (pd.DataFrame) : DataFrame contenant les livres.
        categories_df (pd.DataFrame) : DataFrame contenant les catégories associées aux livres.
        book_id_col (str) : Nom de la colonne identifiant les livres dans les deux DataFrames (par défaut 'book_id').
    Returns:
        pd.DataFrame : DataFrame des livres enrichie avec les catégories.
    """
    books = books.merge(categories_df, on=book_id_col, how='left')
    return books

def add_genres_str_column(books, categories_df, book_id_col="book_id", genres_col="genres_str"):
    """Ajoute une colonne 'categories' contenant les genres des livres sous forme de chaîne de caractères.
    Args:
        books (pd.DataFrame) : DataFrame contenant les informations des livres.
        categories_df (pd.DataFrame) : DataFrame contenant les genres au format dict (colonne 'genres').
        book_id_col (str) : Nom de la colonne identifiant les livres dans les deux DataFrames (par défaut 'book_id').
        genres_col (str) : Nom temporaire de la colonne contenant les genres convertis en chaîne (par défaut 'genres_str').
    Returns:
        pd.DataFrame : DataFrame des livres enrichie d'une colonne 'categories' contenant les genres séparés par des virgules.
    """
    # Copier pour éviter de modifier l'original
    categories_df = categories_df.copy()
    
    # Convertir la colonne 'genres' de string -> dict
    categories_df['genres'] = categories_df['genres'].apply(ast.literal_eval)
    
    # Transformer chaque dict en une chaîne de genres séparés par des virgules
    categories_df[genres_col] = categories_df['genres'].apply(lambda x: ', '.join(x.keys()))
    
    # Supprimer l'ancienne colonne
    categories_df = categories_df.drop(columns=['genres'])

    categories_df.rename(columns={genres_col: "categories"}, inplace=True)
    
    genres_col="categories"
    # Merge avec les livres
    books = books.merge(categories_df, on=book_id_col, how='left')
    
    # Remplacer les NaN par chaîne vide pour les livres sans genres connus
    books[genres_col] = books[genres_col].fillna('')
    
    return books
import json
import psycopg2
from tqdm import tqdm
import re

user='postgres'
password='redsql'

def get_key_books_list(isbn_list):
    """ Renvoie une liste de key de la table 'editions' correspondant aux isbn donnés.
        Arguments :
            isbn_list (str[]) : liste d'isbn
    """
    conn = psycopg2.connect(
        dbname='openlibrary',
        user=user,
        password=password,
        host='localhost',
        port='5432'
    )
    conn.autocommit = True
    cur = conn.cursor()

    sql = """
        SELECT e.key
        FROM editions e
        JOIN edition_isbns ei ON ei.edition_key = e.key
        WHERE ei.isbn = %s
    """

    key_books = []
    for isbn in tqdm(isbn_list, desc='Récupération des clés dans la databse'):
        cur.execute(sql, (isbn,))
        results = cur.fetchall()
        key_books.extend([r[0] for r in results])

    cur.close()
    conn.close()

    return key_books

def update_editions_work_key(keys_list):
    conn = psycopg2.connect(
        dbname='openlibrary',
        user=user,
        password=password,
        host='localhost',
        port='5432'
    )
    conn.autocommit = True
    cur = conn.cursor()

    query = """
        UPDATE editions
        SET work_key = data->'works'->0->>'key'
        WHERE key = %s;
    """

    for key in tqdm(keys_list, desc='Mise à jour de la database avec les clés trouvées'):
        cur.execute(query, (key,))

    cur.close()
    conn.close()

def split_subject_words(subject):
    """ Permet de séparer les mots tout en ignorant la ponctuation et les espaces. 
        On met aussi en minuscule les mots pour éciter la casse.
        Arguments :
            subject (str) : liste de genres retournés lors de la requête sql
    """
    return re.findall(r'\b\w+\b', subject.lower())

def get_used_infos_by_isbn_list(isbn_list):
    conn = psycopg2.connect(
        dbname='openlibrary',
        user=user, 
        password=password, 
        host='localhost',
        port='5432'
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Requête sql permettant de récupérer les genres et descriptions d'un livre via son isbn
    sql = """
        select
        w.data->'description'->>'value' "WorkDescription",
        e.data->>'isbn_13' "ISBN13"
    from editions e
    join edition_isbns ei
        on ei.edition_key = e.key
    join works w
        on w.key = e.work_key
    where ei.isbn = %s
    """

    # Comme on peut avoir plusieurs résultats pour un même isbn, 
    # la requête renverra plusieurs fois des subjects et descriptions (assez souvent identiques)
    # On traite de ce cas juste après
    
    desc_final=[] # liste des descriptions des livres
    isbn_13=[]

    for isbn in tqdm(isbn_list, desc='Récupération des informations supplémentaires'):
        cur.execute(sql, (isbn,))
        results = cur.fetchall()

        # On crée ici des sets pour éviter les doublons lors de l'ajout des descriptions
        description_set = set()
        isbn_13_set = set()

        for r in results:   
            if r[0]:      
                description_set.add(r[0])
            isbn_13_set.add(r[1])

        desc_final.append(list(description_set))
        isbn_13.append(list(isbn_13_set))

    cur.close()
    conn.close()

    return desc_final, isbn_13

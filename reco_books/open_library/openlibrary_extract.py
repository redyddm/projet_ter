import json
import psycopg2
from tqdm import tqdm
import re

from recommandation_de_livres.config import DB_PARAMS

user=DB_PARAMS['user']
password=DB_PARAMS['password']

def get_key_books_list(isbn_list):
    """ Renvoie une liste de key de la table 'editions' correspondant aux isbn donnés.
        Args:
            isbn_list (str[]) : liste d'isbn
        Returns:
            keys_books (str[]) : liste des editions_keys de nos livres
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
    """ Met à jour la base de données avec les clés données. C'est pour éviter de mettre entièrement la base de données à jour.
        Args:
            keys_list (str[]) : liste des clés des données à mettre à jour
        Returns:
            None
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

def get_infos_by_isbn_list(isbn_list):
    """ Permet de récupérer les informations des livres depuis la base de données.
        Args:
            isbn_list (str[]) : Liste d'isbn
        Returns:
            subjects_final (str[][]) : Liste de listes avec les sujets extraits de chaque livre
            desc_final (str[][]) : Liste de listes avec les descriptions extraites de chaque livre
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

    # Requête sql permettant de récupérer les genres et descriptions d'un livre via son isbn
    sql = """
        select
        e.data->>'subjects' "Subjects",
        w.data->'description'->>'value' "WorkDescription"
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
    
    subjects_final= [] # liste de listes qui contiendra les genres recueillis de chaque livre
    desc_final=[] # liste des descriptions des livres
    for isbn in tqdm(isbn_list):
        cur.execute(sql, (isbn,))
        results = cur.fetchall()

        # On crée ici des sets pour éviter les doublons lors de l'ajout des genres ou descriptions
        all_words = set()
        description_set = set()
        for r in results:
            
            if r[0]:
                subjects_list = json.loads(r[0]) 
                for subject in subjects_list:
                    words = split_subject_words(subject) # On sépare les mots des listes de genres qu'on obtient 
                                                         # pour comparer avec les suivants et ne pas ajouter de doublon
                    for w in words:
                        all_words.add(w)

            
            if r[1]:
                description_set.add(r[1])

        subjects_words = list(all_words)
        descriptions = list(description_set)
        subjects_final.append(subjects_words)
        desc_final.append(descriptions)

    cur.close()
    conn.close()

    return subjects_final, desc_final
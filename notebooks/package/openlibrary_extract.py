import json
import psycopg2
from tqdm import tqdm
import re



def get_key_books(isbn_list):
    """ Permet de récupérer une liste de key de la table 'editions' pour chaque isbn de la liste. 
        (C'est une liste de listes de key).
        Arguments :
            isbn_list (str[]) : liste d'isbn
    """
    conn = psycopg2.connect(
        dbname='openlibrary',
        user='postgres',
        password='redsql',
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
    for isbn in tqdm(isbn_list):
        cur.execute(sql, (isbn,))
        results = cur.fetchall()
        print(isbn, results)
        key_books.append([r[0] for r in results])

    cur.close()
    conn.close()

    return key_books

def get_key_books_list(isbn_list):
    """ Renvoie une liste de key de la table 'editions' correspondant aux isbn donnés.
        Arguments :
            isbn_list (str[]) : liste d'isbn
    """
    conn = psycopg2.connect(
        dbname='openlibrary',
        user='postgres',
        password='redsql',
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
    for isbn in tqdm(isbn_list):
        cur.execute(sql, (isbn,))
        results = cur.fetchall()
        key_books.extend([r[0] for r in results])

    cur.close()
    conn.close()

    return key_books

def update_editions_work_key(keys_list):
    conn = psycopg2.connect(
        dbname='openlibrary',
        user='postgres',
        password='redsql',
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

    for key in tqdm(keys_list):
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
    conn = psycopg2.connect(
        dbname='openlibrary',
        user='postgres', 
        password='redsql', 
        host='localhost',
        port='5432'
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Requête sql permettant de récupérer les genres et descriptions d'un livre via son isbn
    sql = """
        select
        e.data->>'subjects' "Subjects",
        e.data->>'isbn13' "ISBN13,
        w.data->'description'->>'value' "WorkDescription",
        e.data->'description'->>'value' "EditionDescription",
        e.data->>'subtitle' "EditionSubtitle",
        w.data->>'subtitle' "WorkSubtitle"
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
    e_descriptions = []
    e_subt=[]
    w_subt=[]

    for isbn in tqdm(isbn_list):
        cur.execute(sql, (isbn,))
        results = cur.fetchall()

        # On crée ici des sets pour éviter les doublons lors de l'ajout des genres ou descriptions
        all_words = set()
        description_set = set()
        e_desc_set = set()
        e_sub = set()
        w_sub = set()

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
            if r[2]:
                e_desc_set.add(r[2])
            if r[3]:
                e_sub.add(r[3])
            if r[4]:
                w_sub.add(r[4])

        subjects_words = list(all_words)
        descriptions = list(description_set)
        subjects_final.append(subjects_words)
        desc_final.append(descriptions)
        e_descriptions.append(list(e_desc_set))
        e_subt.append(list(e_sub))
        w_subt.append(list(w_sub))

    cur.close()
    conn.close()

    return subjects_final, desc_final, e_descriptions, e_subt, w_subt

def get_infos_by_key_list(key_list):
    conn = psycopg2.connect(
        dbname='openlibrary',
        user='postgres', 
        password='redsql', 
        host='localhost',
        port='5432'
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Requête sql permettant de récupérer les genres et descriptions d'un livre via son isbn
    sql = """
        select
        e.data->>'subjects' "Subjects",
        w.data->'description'->>'value' "WorkDescription",
        e.data->'description'->>'value' "EditionDescription",
        e.data->>'subtitle' "EditionSubtitle",
        w.data->>'subtitle' "WorkSubtitle",
        e.data->>'isbn_13' "ISBN13"
    from editions e
    join works w
        on w.key = e.work_key
    where e.key = %s
    """

    # Comme on peut avoir plusieurs résultats pour un même isbn, 
    # la requête renverra plusieurs fois des subjects et descriptions (assez souvent identiques)
    # On traite de ce cas juste après
    
    subjects_final= [] # liste de listes qui contiendra les genres recueillis de chaque livre
    desc_final=[] # liste des descriptions des livres
    e_descriptions = []
    e_subt=[]
    w_subt=[]
    isbn_13=[]

    for key in tqdm(key_list):
        cur.execute(sql, (key,))
        results = cur.fetchall()

        # On crée ici des sets pour éviter les doublons lors de l'ajout des genres ou descriptions
        all_words = set()
        description_set = set()
        e_desc_set = set()
        e_sub = set()
        w_sub = set()

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
            if r[2]:
                e_desc_set.add(r[2])
            if r[3]:
                e_sub.add(r[3])
            if r[4]:
                w_sub.add(r[4])

            if r[5]:
                isbn_13.append(r[5])

        subjects_words = list(all_words)
        descriptions = list(description_set)
        subjects_final.append(subjects_words)
        desc_final.append(descriptions)
        e_descriptions.append(list(e_desc_set))
        e_subt.append(list(e_sub))
        w_subt.append(list(w_sub))

    cur.close()
    conn.close()

    return subjects_final, desc_final, e_descriptions, e_subt, w_subt, isbn_13

def get_desc_by_isbn_list(isbn_list):
    conn = psycopg2.connect(
        dbname='openlibrary',
        user='postgres', 
        password='redsql', 
        host='localhost',
        port='5432'
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Requête sql permettant de récupérer les genres et descriptions d'un livre via son isbn
    sql = """
        select
        w.data->'description'->>'value' "WorkDescription",
        e.data->'description'->>'value' "EditionDescription"
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
    e_descriptions = []

    for isbn in tqdm(isbn_list):
        cur.execute(sql, (isbn,))
        results = cur.fetchall()

        # On crée ici des sets pour éviter les doublons lors de l'ajout des descriptions
        description_set = set()
        e_desc_set = set()

        for r in results:            
            description_set.add(r[0])
            e_desc_set.add(r[1])

        desc_final.append(list(description_set))
        e_descriptions.append(list(e_desc_set))

    cur.close()
    conn.close()

    return desc_final, e_descriptions

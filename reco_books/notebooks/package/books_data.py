import pandas as pd
import re
import glob

chemin = "./datasets/goodreads/books/*.csv"

fichiers_csv = glob.glob(chemin)

#on récupère le nombre du début pour le tri pour le tri
def extraire_debut(fichier):
    # Recherche de "book<nombre>[k]-"
    match = re.search(r'book(\d+)(k)?-', fichier)
    if match:
        nombre = int(match.group(1))
        # si le "k" est présent, on multiplie par 1000
        if match.group(2):
            nombre *= 1000
        return nombre
    return float('inf')

fichiers_csv = sorted(fichiers_csv, key=extraire_debut)

books = pd.concat([pd.read_csv(fichier) for fichier in fichiers_csv], ignore_index=True)
books.head()

chemin_users = "./datasets/goodreads/users/*.csv"

users_csv = glob.glob(chemin_users)
users = pd.concat([pd.read_csv(user) for user in users_csv], ignore_index=True)
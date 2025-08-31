import os
import requests
from lingua import Language, LanguageDetectorBuilder

from recommandation_de_livres.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def save_df_to_csv(df, filepath, index = False):
    """ Sauvegarde le dataFrame en csv dans le chemin filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index, encoding="utf-8")


def save_df_to_pickle(df, filepath):
    """ Sauvegarde le dataFrame en pkl dans le chemin filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_pickle(filepath)

def save_df_to_parquet(df, filepath):
    """ Sauvegarde le dataFrame en parquet dans le chemin filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, engine='pyarrow', index=False, compression='snappy')



def get_cover_url(isbn):
    url = f"https://bookcover.longitood.com/bookcover/{isbn}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("url")
        else:
            return None
    except:
        return None

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

def detect_lang(text, chunk_size=100):
    try:
        lang=detector.detect_language_of(text)
        return lang.iso_code_639_3.name.lower()
    
    except Exception as e:
        return ""

def choose_dataset_interactively():
    """
    Liste les dossiers dans PROCESSED_DATA_DIR et permet à l'utilisateur de choisir.
    Retourne le nom du dossier choisi.
    """
    available_dirs = [d.name for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()]

    if not available_dirs:
        raise ValueError(f"Aucun dataset trouvé dans {PROCESSED_DATA_DIR}")

    print("Choisissez un dataset :")
    for i, name in enumerate(available_dirs, 1):
        print(f"{i}. {name}")

    choice_index = input(f"Votre choix [1-{len(available_dirs)}] : ") or "1"
    try:
        choice_index = int(choice_index)
        if choice_index < 1 or choice_index > len(available_dirs):
            raise ValueError
    except ValueError:
        raise ValueError(f"Choix invalide, entrez un nombre entre 1 et {len(available_dirs)}")

    return available_dirs[choice_index - 1]

def imdb_weighted_rating(v, R, m, C):
    """
    Calcule la note pondérée IMDB pour un item.

    Args:
        v (float) : nombre de votes 
        R (float) : note moyenne de l'item
        m (float) : seuil de popularité
        C (float) : moyenne générale des notes
    """
    return (v / (v + m)) * R + (m / (v + m)) * C
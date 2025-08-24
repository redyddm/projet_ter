import os
import pandas as pd
import requests
from fast_langdetect import detect
from lingua import Language, LanguageDetectorBuilder
from pathlib import Path
import streamlit as st

from recommandation_de_livres.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def save_df_to_csv(df: pd.DataFrame, filepath: str, index: bool = False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index, encoding="utf-8")


def save_df_to_pickle(df: pd.DataFrame, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_pickle(filepath)

def save_df_to_parquet(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    import pandas as pd
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
    
def stars(rating: float, max_stars: int = 5) -> str:
    """Retourne une chaîne d'étoiles ⭐ pour une note donnée."""
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    empty_stars = max_stars - full_stars - int(half_star)

    return "⭐" * full_stars + ("✨" if half_star else "") + "☆" * empty_stars

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

def imdb_weighted_rating(df, m_quantile=0.75):
    """
    Calcule la note pondérée de type IMDB pour un DataFrame books.
    
    df doit contenir :
        - 'average_rating' : note moyenne
        - 'ratings_count'  : nombre de votes
    """
    C = df['average_rating'].mean()
    m = df['ratings_count'].quantile(m_quantile)

    def weighted(row):
        v = row['ratings_count']
        R = row['average_rating']
        return (v / (v + m)) * R + (m / (v + m)) * C

    df['weighted_rating'] = df.apply(weighted, axis=1)
    return df

def choose_dataset_streamlit(raw=True):
    """
    Liste dynamiquement les datasets et fichiers dans RAW_DATA_DIR ou PROCESSED_DATA_DIR
    et retourne le chemin du fichier sélectionné.
    """
    base_path = Path(RAW_DATA_DIR) if raw else Path(PROCESSED_DATA_DIR)

    # Lister les datasets
    datasets = [d for d in base_path.iterdir() if d.is_dir()]
    if not datasets:
        st.error(f"Aucun dataset trouvé dans {base_path}")
        st.stop()

    # Choix du dataset
    selected_dataset = st.selectbox("Sélectionnez un dataset :", [d.name for d in datasets])
    dataset_path = [d for d in datasets if d.name == selected_dataset][0]

    st.session_state['DIR']=selected_dataset
    st.session_state['dataset_path']=dataset_path

    return datasets

def display_files_dataset(raw=True):
    """
    Liste dynamiquement les datasets et fichiers dans RAW_DATA_DIR ou PROCESSED_DATA_DIR
    et retourne le chemin du fichier sélectionné.
    """

    datasets=choose_dataset_streamlit(raw=raw)

    # Choix du dataset
    dataset_path = st.session_state['dataset_path']

    # Lister les fichiers disponibles
    files = list(dataset_path.glob("*.csv")) + list(dataset_path.glob("*.parquet"))
    if not files:
        st.warning("Aucun fichier CSV/Parquet dans ce dataset.")
        st.stop()

    # Choix du fichier
    selected_file = st.selectbox("Sélectionnez un fichier :", [f.name for f in files])

    return dataset_path / selected_file

import os
import pandas as pd
import requests
from fast_langdetect import detect
from lingua import Language, LanguageDetectorBuilder
from pathlib import Path
from recommandation_de_livres.config import PROCESSED_DATA_DIR


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

def stars_html(rating: float, max_stars: int = 5) -> str:
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    empty_stars = max_stars - full_stars - int(half_star)

    html = ""
    html += '<i style="color: gold;">' + "&#9733;" * full_stars + "</i>"
    if half_star:
        html += '<i style="color: gold;">&#189;</i>'  # ou une icône demi-étoile
    html += '<i style="color: lightgray;">' + "&#9733;" * empty_stars + "</i>"
    return html

def stars_final(rating: float, max_stars: int = 5) -> str:
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    empty_stars = max_stars - full_stars - int(half_star)

    stars_html = (
        '<span style="color: gold; font-size: 24px;">' + '★' * full_stars + '</span>' +
        ('<span style="color: gold; font-size: 24px;">☆</span>' if half_star else '') +
        '<span style="color: lightgray; font-size: 24px;">' + '★' * empty_stars + '</span>'
    )

    return stars_html

# ---- CSS pour les étoiles cliquables ----
STAR_STYLE = """
<style>
.star-rating {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 36px;
    cursor: pointer;
}
.star {
    color: #ccc;
    transition: color 0.2s ease-in-out;
}
.star.full {
    color: #FFA41C;
}
.star.half {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
}
.star.half span:first-child {
    color: #FFA41C;
    width: 50%;
    overflow: hidden;
    display: inline-block;
    text-align: left;
    position: absolute;
    left: 0;
}
.star.half span:last-child {
    color: #ccc;
    position: absolute;
}
</style>
"""
# ---- Fonction pour afficher les étoiles dynamiquement ----
def render_stars(rating, max_stars=5):
    stars_html = '<div class="star-rating">'
    for i in range(1, max_stars + 1):
        if rating >= i:
            stars_html += f'<span class="star full">&#9733;</span>'
        elif rating + 0.5 >= i:
            stars_html += '''
            <span class="star half">
                <span>&#9733;</span>
                <span>&#9733;</span>
            </span>
            '''
        else:
            stars_html += '<span class="star">&#9733;</span>'
    stars_html += '</div>'
    return stars_html

# ---- Sélection cliquable (demi-étoiles incluses) ----
"""def star_selector(label, max_stars=5):
    steps = [i * 0.5 for i in range(1, max_stars * 2 + 1)]
    rating = st.radio(label, steps, horizontal=True, format_func=lambda x: f"{x} ★")
    st.markdown(render_stars(rating, max_stars), unsafe_allow_html=True)
    return rating"""

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


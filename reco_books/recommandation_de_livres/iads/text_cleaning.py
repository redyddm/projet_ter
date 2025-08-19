import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords

def nettoyage_texte(text):
    if isinstance(text, str):
        text = " ".join(gensim.utils.simple_preprocess(text))
        text = remove_stopwords(text)
        return text
    return ""

def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r'\(.*?\)', '', title)   # supprime tout ce qui est entre ()
    title = re.sub(r'[^a-z0-9\s]', '', title)  # supprime ponctuation
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def nettoyage_leger(text):
    if isinstance(text, list):
        return [nettoyage_leger(t) for t in text]
    
    if isinstance(text, str):
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\n\r\t]", " ", text)
        text = re.sub(r"\s+", " ", text).strip() 
        return text 
    return ""

def nettoyage_avance(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Minuscule et ponctuation de base
    text = text.lower()
    
    # 2. Supprime ce qui est entre parenthèses
    text = re.sub(r'\(.*?\)', '', text)
    
    # 3. Supprime ponctuation non alphanumérique
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 4. Tokenisation simple + stopwords
    text = " ".join(gensim.utils.simple_preprocess(text))
    
    # 5. Optionnel : enlever les stopwords personnalisés
    text = remove_stopwords(text)
    
    # 6. Nettoyage des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

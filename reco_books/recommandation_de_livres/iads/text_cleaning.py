import re
import html
import gensim
from gensim.parsing.preprocessing import STOPWORDS

book_stopwords = {"a", "an", "the"
}

def remove_custom_stopwords(text):
    return " ".join([w for w in text.lower().split() if w not in book_stopwords])

def nettoyage_balises(text):
    """ On enlève les balises hmtl pour avoir un bon affichage.
        Args:
            text (str) : texte à nettoyer
        Returns:
            text (str) : texte nettoyé
    """

    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def nettoyage_leger(text):
    """ Nettoyage plus léger pour garder les majuscules et ponctuations.
        Args:
            text (str) : texte à nettoyer
        Returns:
            text (str) : texte nettoyé
    """
    if isinstance(text, list):
        return [nettoyage_leger(t) for t in text]
    
    if isinstance(text, str):
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\n\r\t]", " ", text)
        text = re.sub(r"\s+", " ", text).strip() 
        return text 
    return ""

def nettoyage_titre(text):
    """ On nettoie les textes pour le tri.
        Args:
            text (str) : texte à nettoyer
        Returns:
            text (str) : texte nettoyé
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\((?:volume|tome|edition|part|book)\s*\d*\)', '', text)
    text = re.sub(r'\((.*?)\)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def nettoyage_avance(text):
    """ On nettoie les textes de sorte à pouvoir les comparer.
        Args:
            text (str) : texte à nettoyer
        Returns:
            text (str) : texte nettoyé
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\((?:volume|tome|edition|part|book)\s*\d*\)', '', text)
    text = re.sub(r'\((.*?)\)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = " ".join(gensim.utils.simple_preprocess(text, min_len=1))
    text = remove_custom_stopwords(text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
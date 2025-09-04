import re
import html
import gensim
from gensim.parsing.preprocessing import STOPWORDS

book_stopwords = {
    "a", "an", "the",
    "and", "or", "but",
    "of", "in", "on", "to", "for", "by", "with", "at", "from", "about", "as", "into", "like", "through", "after", "over", "between", "out", "against", "during", "without", "before", "under", "around", "among"
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

def nettoyage_avance(text):
    """ On nettoie les texte de sorte à pouvoir les comparer.
        Args:
            text (str) : texte à nettoyer
        Returns:
            text (str) : texte nettoyé
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    print("1", text)
    text = re.sub(r'\(.*?\)', '', text)
    print("2",text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    print("3",text)
    text = " ".join(gensim.utils.simple_preprocess(text))
    print("4",text)
    text = remove_custom_stopwords(text)
    print("5",text)
    text = re.sub(r'\s+', ' ', text).strip()
    print("6",text)
    
    return text

print(nettoyage_avance("You Cannot Be Serious"))


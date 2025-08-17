import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords

def nettoyage_texte(text):
    if isinstance(text, str):
        text = " ".join(gensim.utils.simple_preprocess(text))
        text = remove_stopwords(text)
        return text
    return ""

def nettoyage_leger(text):
    if isinstance(text, list):
        return [nettoyage_leger(t) for t in text]
    
    if isinstance(text, str):
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\n\r\t]", " ", text)
        text = re.sub(r"\s+", " ", text).strip() 
        return text 
    return ""
import re
import ast
import unicodedata


def save_df_to_csv(df, path):
    """ Crée un fichier csv à partir d'un dataframe.
        Arguments :
            df (DataFrame)
            path (str) : chemin du fichier qui sera créé.
    """
    df.to_csv(path, index=False, header=True)

def save_df_to_pickle(df, path):
    """ Crée un fichier pkl à partir d'un dataframe. C'est pour garder le type list des sujets et descriptions.
        Arguments :
            df (DataFrame)
            path (str) : chemin du fichier qui sera créé.
    """
    df.to_pickle(path)
    
def nettoyage_word2vec(text):
    if isinstance(text, str):
        text = text.lower()
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        return re.sub("[^a-z0-9 ]", "", text)
    return ""

def nettoyage_texte(text):
    if isinstance(text, str):
        return re.sub("[^a-zA-Z0-9 ]", "", text) 
    return ""

def nettoyage_leger(text):
    if isinstance(text, str):
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\n\r\t]", " ", text)
        text = re.sub(r"\s+", " ", text).strip() 
        return text 
    return ""

def str_list_to_text(s):
    if isinstance(s, str):
        return s
    else :
        parsed = ast.literal_eval(s)
        return " ".join(parsed)
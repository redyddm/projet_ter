from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import gensim

def get_text_vector(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
    
def get_book_embedding(book_title, model):
    """Génère l'embedding d'un titre de livre selon le modèle fourni.
    Args:
        book_title (str) : Titre du livre.
        model : Modèle utilisé pour générer l'embedding (gensim Word2Vec ou SentenceTransformer).
    Returns:
        np.ndarray : Vecteur embedding du livre.
    Raises:
        ValueError : Si le modèle fourni n'est pas reconnu.
    """
    from sentence_transformers import SentenceTransformer
    if isinstance(model, gensim.models.Word2Vec):
        tokens = gensim.utils.simple_preprocess(book_title)
        vec = get_text_vector(tokens, model).reshape(1, -1)
    elif isinstance(model, SentenceTransformer):
        vec = model.encode([book_title], convert_to_numpy=True)
    else:
        raise ValueError("Modèle non reconnu")
    return vec

def get_book_index(title, books):
    """Récupère l'index d'un livre à partir de son titre.
    Args:
        title (str) : Titre du livre à chercher.
        books (pd.DataFrame) : DataFrame contenant les livres et leurs titres.
    Returns:
        np.ndarray : Indice ou indices du livre correspondant dans le DataFrame.
    """
    index = np.where(books['title'] == title)[0]
    return index

def calcul_cosinus_similarite(books_df, book_title, book_index, book_embedding, embeddings, books_names, k):
    # Calcul des similarités
    similarity = cosine_similarity(book_embedding, embeddings)[0]

    if book_title in books_names:
        similarity[book_index] = 0

    # Top k indices
    top_k_idx = np.argsort(similarity)[-k:][::-1]
    sim_scores = similarity[top_k_idx].copy()
    top_books = books_df.iloc[top_k_idx].copy()

    return top_books, sim_scores

# Pour recherche de titres avec tfidf
def suggest_titles(query, tfidf, tfidf_matrix, books, k=5):
    query_clean = query.lower()
    vec = tfidf.transform([query_clean])
    similarity = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idx = similarity.argsort()[-k:][::-1]
    return books.iloc[top_idx][['title','authors']]

def combine_text(row, cols):
    parts = [str(row.get(col, '')) for col in cols] # fusion des textes des colonnes correspondantes
    return ' '.join(parts)

#--------------FONCTION DE RECOMMANDATION--------------

def recommandation_content_top_k(book_title, embeddings, model, books_df, knn, k=5):
    """Retourne les k livres les plus similaires à un titre donné, selon un modèle, des embeddings et un KNN pré-entraîné.
    Args:
        book_title (str) : Titre du livre de référence.
        embeddings (np.ndarray) : Matrice des embeddings des livres (nb_livres x dimension).
        model : Modèle utilisé pour générer l'embedding si le titre n'est pas dans books_df (Word2Vec, SentenceTransformer, etc.).
        books_df (pd.DataFrame) : DataFrame des livres (doit contenir au moins une colonne 'title').
        knn (sklearn.neighbors.NearestNeighbors) : Modèle KNN pré-entraîné sur les embeddings pour accélérer la recherche de voisins proches.
        k (int) : Nombre de recommandations à retourner.
    Returns:
        (pd.DataFrame, np.ndarray) : 
            - DataFrame contenant les informations des k livres recommandés.
            - Tableau des scores de similarité correspondants (valeurs entre 0 et 1).
    """

    books_names = set(books_df['title'])

    # Embedding du livre donné
    if book_title in books_names:
        book_index = get_book_index(book_title, books_df)
        book_embedding = embeddings[book_index]

        if knn is not None:

            # Calcul des similarités et Top k indices
            distances, top_k_idx = knn.kneighbors(book_embedding)
            top_books = books_df.iloc[top_k_idx[0][1:k+1]].copy()

            sim_scores = 1 - distances
            sim_scores = sim_scores[0][1:k+1]
        else:
            top_books, sim_scores = calcul_cosinus_similarite(books_df, book_title, book_index,
                                                              book_embedding, embeddings, books_names,
                                                              k)       


    # Si le livre n'est pas dans le dataset
    else:
        book_embedding = get_book_embedding(book_title, model)

        # Calcul des similarités
        similarity = cosine_similarity(book_embedding, embeddings)[0]

        # Top k indices
        top_k_idx = np.argsort(similarity)[-k:][::-1]
        sim_scores = similarity[top_k_idx].copy()
        top_books = books_df.iloc[top_k_idx].copy()

    return top_books, sim_scores

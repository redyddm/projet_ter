from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import numpy as np
import gensim


def get_text_vector(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
    
def get_book_embedding(book_title, model):
    if isinstance(model, gensim.models.Word2Vec):
        tokens = gensim.utils.simple_preprocess(book_title)
        vec = get_text_vector(tokens, model).reshape(1, -1)
    elif isinstance(model, SentenceTransformer):
        vec = model.encode([book_title], convert_to_numpy=True)
    else:
        raise ValueError("Modèle non reconnu")
    return vec
    
def recommandation_content_top_k(book_title, embeddings, model, books_df, k=5):

    # Embedding du livre donné
    book_embedding=get_book_embedding(book_title, model)

    # Calcul des similarités
    similarity = cosine_similarity(book_embedding, embeddings)[0]

    # Top k indices
    top_k_idx = np.argsort(similarity)[-k:][::-1]
    top_k_scores = similarity[top_k_idx]

    # Extraire titres et auteurs
    top_books = books_df.iloc[top_k_idx][['title', 'author']].copy()
    top_books['score'] = top_k_scores

    return top_books.reset_index(drop=True)

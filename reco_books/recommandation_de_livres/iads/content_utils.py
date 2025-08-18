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

def get_book_index(title, books):
    index=np.where(books['title']==title)[0]
    return index

def recommandation_content_top_k(book_title, embeddings, model, books_df, k=5):
    books_names=set(books_df['title'])

    # Embedding du livre donné
    if book_title in books_names:
        book_index=get_book_index(book_title, books_df)
        book_embedding=embeddings[book_index]

    else:
        book_embedding=get_book_embedding(book_title, model)
        
    # Calcul des similarités
    similarity = cosine_similarity(book_embedding, embeddings)[0]

    # Top k indices
    top_k_idx = np.argsort(similarity)[-k*2:][::-1]  # prendre un peu plus pour être sûr
    top_k_idx = [i for i in top_k_idx if i != book_index][:k]


    # Extraire titres et auteurs
    top_books = books_df.iloc[top_k_idx][['isbn', 'isbn13', 'title', 'authors']].copy()

    return top_books.reset_index(drop=True)

# Pour recherche de titres avec tfidf
def suggest_titles(query, tfidf, tfidf_matrix, books, k=5):
    query_clean = query.lower()
    vec = tfidf.transform([query_clean])
    similarity = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idx = similarity.argsort()[-k:][::-1]
    return books.iloc[top_idx][['title','authors']]
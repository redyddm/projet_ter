import numpy as np
from tqdm import tqdm
from recommandation_de_livres.iads.content_utils import recommandation_content_top_k
from recommandation_de_livres.iads.collabo_utils import recommandation_collaborative_top_k

def recommandation_hybride_vectorisee(user_id, collaborative_model, content_model, content_df, collaborative_df, books, embeddings, alpha=0.5, k=5, top_k_content=10):
    """
    Recommandation hybride vectorisée.
    - user_id : identifiant utilisateur
    - collaborative_model : modèle collaboratif (SVD, etc.)
    - content_model : modèle content-based
    - content_df : dataframe contenu
    - collaborative_df : dataframe notes
    - embeddings : matrice embeddings livres
    - alpha : poids du score collaboratif
    - k : nombre de recommandations finales
    - top_k_content : nombre de voisins content-based par livre
    """
    # Reco collaborative top-k
    recos_collab, top_k_rating = recommandation_collaborative_top_k(k, user_id, collaborative_model, collaborative_df, books)
    if recos_collab is None or recos_collab.empty:
        return None

    # Standardiser types
    content_df = content_df.copy()
    content_df['item_id'] = content_df['item_id'].astype(str)
    isbn_collab = np.array([str(x) for x in top_k_rating['item_id']])
    score_collab = np.array(top_k_rating['note_predite'].tolist())

    # Initialiser score content global
    score_content_global = np.zeros(len(content_df))

    # Construire un mapping item_id -> index pour vectorisation
    id_to_index = {item: idx for idx, item in enumerate(content_df['item_id'])}

    # Itération seulement sur livres collaboratifs existants dans content_df
    valid_indices = [id_to_index[isbn] for isbn in isbn_collab if isbn in id_to_index]
    valid_scores = [score_collab[i] for i, isbn in enumerate(isbn_collab) if isbn in id_to_index]

    for ref_index, s_collab in zip(valid_indices, valid_scores):
        # Content-based top-k pour chaque livre
        top_books, sim = recommandation_content_top_k(content_df.iloc[ref_index]['title'], embeddings, content_model, content_df, top_k_content)
        # Combinaison vectorisée : ajouter à score_content_global
        score_content_global[top_books.index] += sim * s_collab
        score_content_global[ref_index] = 0  # Exclure le livre lui-même
        #score_content_global += sim * s_collab

    # Normalisation content-based
    score_content_norm = (score_content_global - score_content_global.min()) / (score_content_global.max() - score_content_global.min() + 1e-8)

    # Score collaboratif global vectorisé
    score_collab_global = np.zeros(len(content_df))
    indices_in_df = [id_to_index[isbn] for isbn in isbn_collab if isbn in id_to_index]
    score_collab_global[indices_in_df] = [score_collab[i] for i, isbn in enumerate(isbn_collab) if isbn in id_to_index]

    score_collab_norm = (score_collab_global - score_collab_global.min()) / (score_collab_global.max() - score_collab_global.min() + 1e-8)

    # Score final hybride
    score_final = alpha * score_collab_norm + (1 - alpha) * score_content_norm

    # Résultat final
    result_df = content_df.copy()
    result_df['score_hybride'] = score_final

    # Exclure livres déjà lus
    result_df = result_df[~result_df['item_id'].isin(isbn_collab)]
    result_df = result_df.sort_values('score_hybride', ascending=False)

    return result_df.head(k)

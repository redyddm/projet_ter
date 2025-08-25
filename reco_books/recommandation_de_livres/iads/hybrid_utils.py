import numpy as np
from recommandation_de_livres.iads.content_utils import recommandation_content_top_k
from recommandation_de_livres.iads.collabo_utils import recommandation_collaborative_top_k

def recommandation_hybride_vectorisee(user_id, collaborative_model, content_model,
                                      content_df, collaborative_df, books,
                                      embeddings, knn=None,
                                      alpha=0.5, k=5, top_k_content=10):
    """
    Recommandation hybride vectorisée utilisant collaboratif + contenu (Word2Vec / SBERT) avec KNN optionnel.
    
    Args:
        user_id : identifiant utilisateur
        collaborative_model : modèle collaboratif (SVD, etc.)
        content_model : modèle content-based
        content_df : dataframe des livres avec info de contenu
        collaborative_df : dataframe des notes utilisateurs
        books : dataframe complet des livres
        embeddings : matrice embeddings content-based
        knn : modèle KNN pré-entraîné pour accélérer les recherches content-based (optionnel)
        alpha : poids du score collaboratif (0=contenu, 1=collaboratif)
        k : nombre de recommandations finales
        top_k_content : nombre de voisins content-based par livre collaboratif
        
    Returns:
        pd.DataFrame : top k livres recommandés avec colonne 'score_hybride'
    """

    # --- Reco collaborative top-k ---
    recos_collab, top_k_rating = recommandation_collaborative_top_k(
        k=k, user_id=user_id, model=collaborative_model,
        ratings=collaborative_df, books=books
    )
    if recos_collab is None or recos_collab.empty:
        return None

    # --- Préparer data ---
    content_df = content_df.copy()
    content_df['item_id'] = content_df['item_id'].astype(str)
    isbn_collab = np.array([str(x) for x in top_k_rating['item_id']])
    score_collab = np.array(top_k_rating['note_predite'].tolist())

    id_to_index = {item: idx for idx, item in enumerate(content_df['item_id'])}
    score_content_global = np.zeros(len(content_df))

    # --- Calcul content-based pour chaque livre collaboratif ---
    for i, isbn in enumerate(isbn_collab):
        if isbn not in id_to_index:
            continue
        ref_index = id_to_index[isbn]
        s_collab = score_collab[i]

        # Utilisation KNN si fourni, sinon calcul classique
        top_books, sim = recommandation_content_top_k(
            content_df.iloc[ref_index]['title'],
            embeddings,
            content_model,
            content_df,
            knn=knn,
            k=top_k_content
        )

        # Normaliser les similarités par livre
        sim_norm = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        score_content_global[top_books.index] += sim_norm * s_collab

    # --- Normalisation content et collaboratif ---
    score_content_norm = (score_content_global - score_content_global.min()) / \
                         (score_content_global.max() - score_content_global.min() + 1e-8)

    score_collab_global = np.zeros(len(content_df))
    indices_in_df = [id_to_index[isbn] for isbn in isbn_collab if isbn in id_to_index]
    score_collab_global[indices_in_df] = [score_collab[i] for i, isbn in enumerate(isbn_collab) if isbn in id_to_index]
    score_collab_norm = (score_collab_global - score_collab_global.min()) / \
                        (score_collab_global.max() - score_collab_global.min() + 1e-8)

    # --- Score final hybride ---
    score_final = alpha * score_collab_norm + (1 - alpha) * score_content_norm

    result_df = content_df.copy()
    result_df['score_hybride'] = score_final

    # Exclure livres déjà vus
    result_df = result_df[~result_df['item_id'].isin(isbn_collab)]
    result_df = result_df.sort_values('score_hybride', ascending=False)

    return result_df.head(k)

import numpy as np
from recommandation_de_livres.iads.content_utils import recommandation_content_top_k
from recommandation_de_livres.iads.collabo_utils import recommandation_collaborative_top_k

def recommandation_hybride_mmr_knn(user_id, collaborative_model, content_model, content_df,
                                   collaborative_df, books, embeddings, knn,
                                   alpha=0.5, k=5, top_k_content=10):
    """
    Recommandation hybride MMR/KNN.
    - user_id : identifiant utilisateur
    - collaborative_model : modèle collaboratif (SVD, etc.)
    - content_model : modèle content-based
    - content_df : DataFrame contenu
    - collaborative_df : DataFrame notes
    - embeddings : matrice embeddings livres
    - knn : modèle NearestNeighbors pré-entraîné
    - alpha : poids du score collaboratif (0=contenu uniquement, 1=collaboratif uniquement)
    - k : nombre de recommandations finales
    - top_k_content : nombre de voisins content-based par livre
    """

    # --- Reco collaborative top-k ---
    recos_collab, top_k_rating = recommandation_collaborative_top_k(
        k, user_id, collaborative_model, collaborative_df, books
    )
    if recos_collab is None or recos_collab.empty:
        return None

    # Préparer dataframe
    content_df = content_df.copy()
    content_df['item_id'] = content_df['item_id'].astype(str)
    isbn_collab = np.array([str(x) for x in top_k_rating['item_id']])
    score_collab = np.array(top_k_rating['note_predite'].tolist())

    # --- Mapping item_id -> index ---
    id_to_index = {item: idx for idx, item in enumerate(content_df['item_id'])}

    # --- Score content global initialisé à zéro ---
    score_content_global = np.zeros(len(content_df))

    # Itération sur livres collaboratifs existants
    for i, isbn in enumerate(isbn_collab):
        if isbn not in id_to_index:
            continue
        ref_index = id_to_index[isbn]
        # Content-based top-k pour ce livre
        top_books, sim = recommandation_content_top_k(
            content_df.iloc[ref_index]['title'], embeddings, content_model, content_df,
            knn=knn, k=top_k_content
        )
        # Ajouter similarité pure, sans pondérer par score_collab
        score_content_global[top_books.index] += sim
        # Exclure le livre lui-même
        score_content_global[ref_index] = 0

    # --- Normalisation ---
    if score_content_global.max() > 0:
        score_content_norm = (score_content_global - score_content_global.min()) / \
                             (score_content_global.max() - score_content_global.min() + 1e-8)
    else:
        score_content_norm = score_content_global

    # Score collaboratif global
    score_collab_global = np.zeros(len(content_df))
    for i, isbn in enumerate(isbn_collab):
        if isbn in id_to_index:
            score_collab_global[id_to_index[isbn]] = score_collab[i]

    # Normalisation collaboratif
    if score_collab_global.max() > 0:
        score_collab_norm = (score_collab_global - score_collab_global.min()) / \
                            (score_collab_global.max() - score_collab_global.min() + 1e-8)
    else:
        score_collab_norm = score_collab_global

    # --- Score final hybride ---
    score_final = alpha * score_collab_norm + (1 - alpha) * score_content_norm

    # --- Résultat final ---
    result_df = content_df.copy()
    result_df['score_hybride'] = score_final

    # Exclure livres déjà vus
    result_df = result_df[~result_df['item_id'].isin(isbn_collab)]
    result_df = result_df.sort_values('score_hybride', ascending=False)

    return result_df.head(k)

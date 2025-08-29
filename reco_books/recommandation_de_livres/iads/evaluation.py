import numpy as np

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Calcule la précision et le rappel moyens @K pour un ensemble de prédictions générées avec Surprise.

    Args:
        predictions (list of tuples) : Liste de prédictions au format Surprise `(uid, iid, true_r, est, details)`
        k (int, optional) : Nombre d’items à considérer pour le calcul de Precision@K et Recall@K. Defaults to 10.
        threshold (float, optional) : Seuil au-dessus duquel une note est considérée comme positive. Defaults to 3.5.

    Returns:
        tuple: 
            - precision_mean (float) : Moyenne des Precision@K pour tous les utilisateurs
            - recall_mean (float) : Moyenne des Recall@K pour tous les utilisateurs
    """
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))

    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)  # Tri par score estimé décroissant
        n_rel = sum(true_r >= threshold for (_, true_r) in ratings)
        n_rec_k = sum(est >= threshold for (est, _) in ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold) and (est >= threshold) for (est, true_r) in ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return np.mean(list(precisions.values())), np.mean(list(recalls.values()))
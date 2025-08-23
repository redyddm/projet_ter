import matplotlib.pyplot as plt
from collections import defaultdict

def precision_recall_at_k(predictions, k=5, threshold=3.5):
    """Calcul Precision@k et Recall@k pour un modèle."""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k_preds = user_ratings[:k]
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k_preds)
        n_rel_and_rec_k = sum((true_r >= threshold and est >= threshold) for (est, true_r) in top_k_preds)
        precisions.append(n_rel_and_rec_k / max(n_rec_k, 1))
        recalls.append(n_rel_and_rec_k / max(n_rel, 1))
    
    return sum(precisions)/len(precisions), sum(recalls)/len(recalls)

def plot_precision_recall_separate(preds_svd, preds_nmf, max_k=20, threshold=3.5):
    """
    Trace Precision@k et Recall@k pour SVD et NMF pour k=1..max_k
    """
    k_values = list(range(1, max_k+1))
    svd_precisions, svd_recalls = [], []
    nmf_precisions, nmf_recalls = [], []

    for k in k_values:
        p, r = precision_recall_at_k(preds_svd, k=k, threshold=threshold)
        svd_precisions.append(p)
        svd_recalls.append(r)

        p, r = precision_recall_at_k(preds_nmf, k=k, threshold=threshold)
        nmf_precisions.append(p)
        nmf_recalls.append(r)

    # --- Figure Precision ---
    fig_prec, ax_prec = plt.subplots()
    ax_prec.plot(k_values, svd_precisions, label="SVD", marker='o')
    ax_prec.plot(k_values, nmf_precisions, label="NMF", marker='o')
    ax_prec.set_xlabel("k")
    ax_prec.set_ylabel("Precision@k")
    ax_prec.set_title("Precision@k SVD vs NMF")
    ax_prec.legend()
    ax_prec.grid(True)

    # --- Figure Recall ---
    fig_rec, ax_rec = plt.subplots()
    ax_rec.plot(k_values, svd_recalls, label="SVD", marker='o')
    ax_rec.plot(k_values, nmf_recalls, label="NMF", marker='o')
    ax_rec.set_xlabel("k")
    ax_rec.set_ylabel("Recall@k")
    ax_rec.set_title("Recall@k SVD vs NMF")
    ax_rec.legend()
    ax_rec.grid(True)

    return fig_prec, fig_rec

import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import KFold
from recommandation_de_livres.loaders.load_data import load_parquet
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
import pickle
from sentence_transformers import SentenceTransformer
from recommandation_de_livres.iads.hybrid_utils import recommandation_hybride_vectorisee
from recommandation_de_livres.iads.collabo_utils import recommandation_collaborative_top_k

# --- Paramètres globaux ---
DIR = st.session_state.get("DIR", None)
if not DIR:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à l'accueil.")
    st.stop()

st.title("⚙️ Évaluation avancée des modèles collaboratifs et hybrides")

# --- Chargement des fichiers ---
st.info("Chargement des données...")
collab_df = load_parquet(PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet")
content_df = load_parquet(PROCESSED_DATA_DIR / DIR / "content_dataset.parquet")
embeddings = np.load(PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy")
with open(MODELS_DIR / DIR / "svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)
sbert_model = SentenceTransformer(str(MODELS_DIR / DIR / "sbert_model"))
st.success(f"Datasets chargés : {len(collab_df)} notes, {len(content_df)} livres.")

# --- Paramètres ---
k_top = st.slider("K pour Precision@K et Recall@K", 1, 50, 10)
threshold = st.slider("Seuil d'une recommandation 'pertinente'", 1.0, 5.0, 4.0, 0.5)
alpha_hybrid = st.slider("Poids collaboratif dans l'hybride (alpha)", 0.0, 1.0, 0.5, 0.05)
top_k_content = st.slider("Nombre de voisins content-based par livre", 5, 50, 30, 5)

# --- Precision/Recall function ---
def precision_recall_at_k(preds, k=10, threshold=3.5):
    user_est_true = {}
    for uid, iid, true_r, est, _ in preds:
        user_est_true.setdefault(uid, []).append((est, true_r))
    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for (_, true_r) in ratings)
        n_rec_k = sum(est >= threshold for (est, _) in ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold) and (est >= threshold) for (est, true_r) in ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    return np.mean(list(precisions.values())), np.mean(list(recalls.values()))

# --- Cross-validation ---
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(collab_df[['user_id','item_id','rating']], reader)
kf = KFold(n_splits=3)  # pour test rapide
results = {"SVD":{}, "Hybride":{}}

for fold, (trainset, testset) in enumerate(kf.split(data), 1):
    st.info(f"✅ Fold {fold} en cours...")
    
    # --- SVD ---
    svd_model.fit(trainset)
    predictions = svd_model.test(testset)
    rmse = np.sqrt(np.mean([(pred.est - pred.r_ui)**2 for pred in predictions]))
    mae = np.mean([abs(pred.est - pred.r_ui) for pred in predictions])
    prec, rec = precision_recall_at_k(predictions, k=k_top, threshold=threshold)
    all_items = collab_df['item_id'].unique()
    recommended_items = set(pred.iid for pred in predictions if pred.est >= threshold)
    catalog_cov = len(recommended_items) / len(all_items)
    # Stocker
    for key, val in zip(["RMSE","MAE","Precision@K","Recall@K","CatalogCoverage"], [rmse, mae, prec, rec, catalog_cov]):
        results["SVD"].setdefault(key, []).append(val)

    # --- Hybride ---
    user_ids = set([pred.uid for pred in predictions])
    recommended_hybrid = set()
    for uid in user_ids:
        rec_h = recommandation_hybride_vectorisee(
            user_id=uid,
            collaborative_model=svd_model,
            content_model=sbert_model,
            content_df=content_df,
            collaborative_df=collab_df,
            books=collab_df,
            embeddings=embeddings,
            alpha=alpha_hybrid,
            k=k_top,
            top_k_content=top_k_content
        )
        if rec_h is not None:
            recommended_hybrid.update(rec_h['item_id'].tolist())
    catalog_cov_h = len(recommended_hybrid) / len(all_items)
    results["Hybride"].setdefault("CatalogCoverage", []).append(catalog_cov_h)
    st.success(f"Fold {fold} terminé ✅")

# --- Résultats finaux ---
st.subheader("📊 Résultats moyens par modèle")
summary = {}
for model in results:
    summary[model] = {metric: np.mean(vals) for metric, vals in results[model].items()}
st.dataframe(pd.DataFrame(summary).T)

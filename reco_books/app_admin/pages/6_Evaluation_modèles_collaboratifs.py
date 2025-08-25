import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from surprise import SVD, NMF, Dataset, Reader
from surprise.model_selection import cross_validate, KFold
from recommandation_de_livres.config import PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_parquet

# --- Fonctions pour Precision@K et Recall@K ---
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Calcule Precision@K et Recall@K pour un ensemble de prédictions Surprise"""
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

# --- TITRE ---
st.title("⚙️ Évaluation avancée des modèles collaboratifs (SVD & NMF)")

# --- Choix du dataset ---
DIR = st.session_state.get("DIR", None)
if not DIR:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à l'accueil.")
    st.stop()

features_path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet"
if not features_path.exists():
    st.error(f"Fichier introuvable : {features_path}")
    st.stop()

# --- Chargement des données ---
st.info("Chargement du dataset collaboratif...")
collaborative_df = load_parquet(features_path)
st.write(f"📊 Dataset chargé avec **{len(collaborative_df)} notes**.")
st.write(f"Nombre de livres uniques : {len(collaborative_df['item_id'].unique())}")

# --- Échantillonnage optionnel ---
sample_size = st.number_input("Taille de l'échantillon (0 = tout le dataset)", min_value=0, max_value=len(collaborative_df), value=50000, step=1000)
if sample_size > 0 and sample_size < len(collaborative_df):
    collaborative_df = collaborative_df.sample(sample_size, random_state=42)
    st.success(f"Utilisation d'un échantillon de **{len(collaborative_df)}** notes.")

# --- Paramètres globaux ---
min_rating = st.number_input("Note minimale", min_value=1.0, max_value=10.0, value=1.0, step=0.5)
max_rating = st.number_input("Note maximale", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
if min_rating >= max_rating:
    st.error("⚠️ La note minimale doit être inférieure à la note maximale.")
    st.stop()
rating_scale = (min_rating, max_rating)

k_top = st.slider("K pour Precision@K et Recall@K", min_value=1, max_value=50, value=10)
threshold = st.slider("Seuil d'une recommandation 'pertinente'", min_value=float(min_rating), max_value=float(max_rating), value=4.0, step=0.5)

# --- Paramètres spécifiques SVD ---
tab1, tab2 = st.tabs(['Paramètres SVD', 'Paramètres NMF'])

with tab1:
    st.subheader("⚙️ Paramètres du modèle SVD")
    n_factors_svd = st.slider("Nombre de facteurs latents", min_value=0, max_value=300, value=100, step=5, key="svd fact")
    n_epochs_svd = st.slider("Nombre d'itérations", min_value=10, max_value=200, value=50, step=10, key="svd epochs")
    lr_all_svd = st.number_input("Taux d'apprentissage", min_value=0.001, max_value=0.1, value=0.005, step=0.001)
    reg_all_svd = st.number_input("Régularisation", min_value=0.001, max_value=0.1, value=0.02, step=0.001)

with tab2:
    st.subheader("⚙️ Paramètres du modèle NMF")
    n_factors_nmf = st.slider("Nombre de facteurs latents", min_value=0, max_value=300, value=100, step=5, key="nmf fact")
    n_epochs_nmf = st.slider("Nombre d'itérations", min_value=10, max_value=200, value=50, step=10, key="nmf epochs")
    reg_pu_nmf = st.number_input("Régularisation utilisateurs", min_value=0.001, max_value=0.1, value=0.06, step=0.001)
    reg_qi_nmf = st.number_input("Régularisation items", min_value=0.001, max_value=0.1, value=0.06, step=0.001)

# --- Bouton d'évaluation ---
if st.button("🚀 Lancer la cross-validation complète"):
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(collaborative_df[['user_id', 'item_id', 'rating']], reader)

    results = {}
    kf = KFold(n_splits=5)

    algos = {
        "SVD": SVD(n_factors=n_factors_svd, n_epochs=n_epochs_svd, lr_all=lr_all_svd, reg_all=reg_all_svd),
        "NMF": NMF(n_factors=n_factors_nmf, n_epochs=n_epochs_nmf, reg_pu=reg_pu_nmf, reg_qi=reg_qi_nmf)
    }

    for model_name, algo in algos.items():
        with st.spinner(f"Évaluation du modèle {model_name}..."):
            rmse_scores, mae_scores, precisions, recalls, catalog_coverages = [], [], [], [], []

            for trainset, testset in kf.split(data):
                algo.fit(trainset)
                predictions = algo.test(testset)

                # RMSE & MAE
                rmse = np.sqrt(np.mean([(pred.est - pred.r_ui) ** 2 for pred in predictions]))
                mae = np.mean([abs(pred.est - pred.r_ui) for pred in predictions])
                rmse_scores.append(rmse)
                mae_scores.append(mae)

                # Precision@K & Recall@K
                prec, rec = precision_recall_at_k(predictions, k=k_top, threshold=threshold)
                precisions.append(prec)
                recalls.append(rec)

                # Catalog Coverage
                all_items = collaborative_df['item_id'].unique()
                recommended_items = set(pred.iid for pred in predictions if pred.est >= threshold)
                catalog_cov = len(recommended_items) / len(all_items)
                catalog_coverages.append(catalog_cov)

            results[model_name] = {
                "RMSE": np.mean(rmse_scores),
                "MAE": np.mean(mae_scores),
                "Precision@K": np.mean(precisions),
                "Recall@K": np.mean(recalls),
                "CatalogCoverage": np.mean(catalog_coverages)
            }

    # --- Résultats ---
    st.subheader("📊 Résultats comparatifs")
    summary = pd.DataFrame(results).T
    styled_summary = summary.style.highlight_max(
        subset=["Precision@K", "Recall@K", "CatalogCoverage"], color="lightgreen", axis=0
    ).highlight_min(
        subset=["RMSE", "MAE"], color="lightblue", axis=0
    )
    st.dataframe(styled_summary)
    st.success("Évaluation terminée ✅")

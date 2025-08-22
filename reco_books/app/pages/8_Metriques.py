import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR

st.title("📊 Évaluation du modèle SVD")

# --- Choix du dataset ---
choice = st.radio("Choix du dataset :", ["Recommender (1)", "Goodreads (2)"], index=1)

if choice.startswith("Recommender"):
    DIR = "recommender"
    rating_scale = (1, 10)
    id_col = "ISBN"
else:
    DIR = "goodreads"
    rating_scale = (1, 5)
    id_col = "book_id"

features_path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet"

# --- Chargement des données ---
collab_df = pd.read_parquet(features_path)

# --- Hyperparamètres ---
st.subheader("🔧 Hyperparamètres du modèle")
n_factors = st.number_input("Nombre de facteurs latents", min_value=10, max_value=500, value=50, step=10)
n_epochs = st.number_input("Nombre d'epochs", min_value=5, max_value=200, value=50, step=5)
lr_all = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.002, step=0.001, format="%.4f")
reg_all = st.number_input("Régularisation", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.4f")

# --- Création du dataset surprise ---
reader = Reader(rating_scale=rating_scale)
data = Dataset.load_from_df(collab_df[['user_id', id_col, 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# --- Entraînement du modèle ---
if st.button("🚀 Lancer l'entraînement et les prédictions"):
    svd = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    with st.spinner("Entraînement et prédiction... ⏳"):
        svd.fit(trainset)
        preds = svd.test(testset)
        st.session_state["predictions"] = preds
    st.success("✅ Notes prédites")

# --- Évaluation classique ---
if "predictions" in st.session_state:
    preds = st.session_state["predictions"]

    if st.button("Afficher le RMSE et le MAE"):
        rmse = accuracy.rmse(preds, verbose=False)
        mae = accuracy.mae(preds, verbose=False)

        st.subheader("📌 Métriques classiques")
        st.write(f"RMSE : {rmse:.4f}")
        st.write(f"MAE  : {mae:.4f}")

    # --- Évaluation top-k ---
    st.subheader("⭐ Precision@k / Recall@k")
    k = st.slider("Top-k pour Precision/Recall", min_value=1, max_value=20, value=5)

    def precision_recall_at_k(predictions, k=5, threshold=3.5):
        from collections import defaultdict
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

    precision_k, recall_k = precision_recall_at_k(preds, k=k)
    st.write(f"Precision@{k} : {precision_k:.4f}")
    st.write(f"Recall@{k}    : {recall_k:.4f}")

else:
    st.info("ℹ️ Lancez l'entraînement pour pouvoir afficher les métriques.")
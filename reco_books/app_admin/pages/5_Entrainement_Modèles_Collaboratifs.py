import streamlit as st
import pandas as pd
from surprise import SVD, NMF, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pickle
from collections import defaultdict
from pathlib import Path
from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR

st.title("⚙️ Entraînement et comparaison SVD / NMF")

# --- Choix du dataset ---
choice = st.selectbox("Choix du dataset :", ["Recommender (1)", "Goodreads (2)"], index=1)

if choice.startswith("Recommender"):
    DIR = "recommender"
    rating_scale = (1, 10)
    id_col = "ISBN"
else:
    DIR = "goodreads"
    rating_scale = (1, 5)
    id_col = "book_id"

# --- Chargement des données ---
features_path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet"
collab_df = pd.read_parquet(features_path)
st.write(f"📊 {len(collab_df)} notes chargées.")

# --- Fonction Precision@k / Recall@k ---
def precision_recall_at_k(predictions, k=5, threshold=3.5):
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

# --- Tabs pour SVD et NMF ---
tab1, tab2 = st.tabs(["SVD", "NMF"])

# --- Tab SVD ---
with tab1:
    st.subheader("🔧 Hyperparamètres SVD")
    n_factors_svd = st.number_input("Nombre de facteurs latents", min_value=10, max_value=500, value=50, step=10, key="svd_n_factors")
    n_epochs_svd = st.number_input("Nombre d'epochs", min_value=5, max_value=200, value=50, step=5, key="svd_n_epochs")
    lr_all_svd = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.002, step=0.001, format="%.4f", key="svd_lr")
    reg_all_svd = st.number_input("Régularisation", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.4f", key="svd_reg")
    k_svd = st.slider("Top-k pour Precision/Recall", min_value=1, max_value=20, value=5, key="svd_k")

    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(collab_df[['user_id', id_col, 'rating']], reader)
    trainset_svd, testset_svd = train_test_split(data, test_size=0.2)

    if st.button("🚀 Entraîner SVD", key="svd_train"):
        svd = SVD(n_factors=n_factors_svd, n_epochs=n_epochs_svd, lr_all=lr_all_svd, reg_all=reg_all_svd)
        with st.spinner("Entraînement SVD... ⏳"):
            svd.fit(trainset_svd)
            preds_svd = svd.test(testset_svd)
            rmse = accuracy.rmse(preds_svd, verbose=False)
            mae = accuracy.mae(preds_svd, verbose=False)
            precision, recall = precision_recall_at_k(preds_svd, k=k_svd)
            st.session_state["svd_metrics"] = {
                "RMSE": rmse,
                "MAE": mae,
                f"Precision@{k_svd}": precision,
                f"Recall@{k_svd}": recall
            }
        st.success("✅ SVD entraîné et métriques calculées")
    
    if "svd_metrics" in st.session_state:
        st.subheader("📌 Métriques SVD")
        st.table(st.session_state["svd_metrics"])
        if st.button("💾 Sauvegarder SVD final"):
            model_path = MODELS_DIR / DIR / "svd_model_final.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(svd, open(model_path, "wb"))
            st.success(f"SVD sauvegardé dans {model_path}")

# --- Tab NMF ---
with tab2:
    st.subheader("🔧 Hyperparamètres NMF")
    n_factors_nmf = st.number_input("Nombre de facteurs latents", min_value=5, max_value=500, value=50, step=5, key="nmf_n_factors")
    n_epochs_nmf = st.number_input("Nombre d'epochs", min_value=5, max_value=200, value=50, step=5, key="nmf_n_epochs")
    reg_pu_nmf = st.number_input("Régularisation utilisateurs", min_value=0.0001, max_value=1.0, value=0.06, step=0.01, format="%.4f", key="nmf_reg_pu")
    reg_qi_nmf = st.number_input("Régularisation items", min_value=0.0001, max_value=1.0, value=0.06, step=0.01, format="%.4f", key="nmf_reg_qi")
    k_nmf = st.slider("Top-k pour Precision/Recall", min_value=1, max_value=20, value=5, key="nmf_k")

    trainset_nmf, testset_nmf = train_test_split(data, test_size=0.2)

    if st.button("🚀 Entraîner NMF", key="nmf_train"):
        nmf = NMF(n_factors=n_factors_nmf, n_epochs=n_epochs_nmf, reg_pu=reg_pu_nmf, reg_qi=reg_qi_nmf)
        with st.spinner("Entraînement NMF... ⏳"):
            nmf.fit(trainset_nmf)
            preds_nmf = nmf.test(testset_nmf)
            rmse = accuracy.rmse(preds_nmf, verbose=False)
            mae = accuracy.mae(preds_nmf, verbose=False)
            precision, recall = precision_recall_at_k(preds_nmf, k=k_nmf)
            st.session_state["nmf_metrics"] = {
                "RMSE": rmse,
                "MAE": mae,
                f"Precision@{k_nmf}": precision,
                f"Recall@{k_nmf}": recall
            }
        st.success("✅ NMF entraîné et métriques calculées")
    
    if "nmf_metrics" in st.session_state:
        st.subheader("📌 Métriques NMF")
        st.table(st.session_state["nmf_metrics"])
        if st.button("💾 Sauvegarder NMF final"):
            model_path = MODELS_DIR / DIR / "nmf_model_final.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(nmf, open(model_path, "wb"))
            st.success(f"NMF sauvegardé dans {model_path}")

import streamlit as st
import pandas as pd
from surprise import SVD, NMF, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pickle
from collections import defaultdict
from pathlib import Path

from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.plots.graphes import plot_precision_recall_separate


if not st.session_state.get("logged_in"):
    st.warning("Vous devez vous connecter pour accéder à cette page.")
    st.stop()

if st.session_state.get("is_admin") is False:
    st.warning("Cette page est réservée aux admins.")
    st.stop()

st.title("⚙️ Entraînement et comparaison SVD / NMF")

# --- Choix du dataset ---
choice = st.selectbox("Choix du dataset :", ["Recommender", "Goodreads", "Fusion"], index=0)

if choice.startswith("Recommender"):
    DIR = "recommender"
    rating_scale = (1, 10)
elif choice.startswith("Goodreads"):
    DIR = "goodreads"
    rating_scale = (1, 5)

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

# --- Tabs principales ---
tab_hyper_svd, tab_hyper_nmf, tab_train = st.tabs(["SVD Hyperparamètres", "NMF Hyperparamètres", "Entraînement & Métriques"])

# --- Tab SVD Hyperparamètres ---
with tab_hyper_svd:
    st.subheader("🔧 Hyperparamètres SVD")
    n_factors_svd = st.number_input("Nombre de facteurs latents SVD", min_value=10, max_value=500, value=50, step=10, key="svd_n_factors")
    n_epochs_svd = st.number_input("Nombre d'epochs SVD", min_value=5, max_value=200, value=50, step=5, key="svd_n_epochs")
    lr_all_svd = st.number_input("Learning rate SVD", min_value=0.0001, max_value=0.1, value=0.002, step=0.001, format="%.4f", key="svd_lr")
    reg_all_svd = st.number_input("Régularisation SVD", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.4f", key="svd_reg")
    k_svd = st.slider("Top-k pour Precision/Recall SVD", min_value=1, max_value=20, value=5, key="svd_k")

# --- Tab NMF Hyperparamètres ---
with tab_hyper_nmf:
    st.subheader("🔧 Hyperparamètres NMF")
    n_factors_nmf = st.number_input("Nombre de facteurs latents NMF", min_value=5, max_value=500, value=50, step=5, key="nmf_n_factors")
    n_epochs_nmf = st.number_input("Nombre d'epochs NMF", min_value=5, max_value=200, value=50, step=5, key="nmf_n_epochs")
    reg_pu_nmf = st.number_input("Régularisation utilisateurs NMF", min_value=0.0001, max_value=1.0, value=0.06, step=0.01, format="%.4f", key="nmf_reg_pu")
    reg_qi_nmf = st.number_input("Régularisation items NMF", min_value=0.0001, max_value=1.0, value=0.06, step=0.01, format="%.4f", key="nmf_reg_qi")
    k_nmf = st.slider("Top-k pour Precision/Recall NMF", min_value=1, max_value=20, value=5, key="nmf_k")

# --- Tab Entraînement & Métriques ---
with tab_train:
    st.subheader("🚀 Entraînement et visualisation")
    
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(collab_df[['user_id', 'item_id', 'rating']], reader)
    trainset_svd, testset_svd = train_test_split(data, test_size=0.2)
    trainset_nmf, testset_nmf = train_test_split(data, test_size=0.2)

    if st.button("Entraîner SVD et NMF", key="train_both"):
        # --- SVD ---
        svd = SVD(n_factors=n_factors_svd, n_epochs=n_epochs_svd, lr_all=lr_all_svd, reg_all=reg_all_svd)
        with st.spinner("Entraînement SVD... ⏳"):
            svd.fit(trainset_svd)
            preds_svd = svd.test(testset_svd)
            rmse_svd = accuracy.rmse(preds_svd, verbose=False)
            mae_svd = accuracy.mae(preds_svd, verbose=False)
            precision_svd, recall_svd = precision_recall_at_k(preds_svd, k=k_svd)
            st.session_state["svd_metrics"] = {"RMSE": rmse_svd, "MAE": mae_svd, "Precision": precision_svd, "Recall": recall_svd}

        # --- NMF ---
        nmf = NMF(n_factors=n_factors_nmf, n_epochs=n_epochs_nmf, reg_pu=reg_pu_nmf, reg_qi=reg_qi_nmf)
        with st.spinner("Entraînement NMF... ⏳"):
            nmf.fit(trainset_nmf)
            preds_nmf = nmf.test(testset_nmf)
            rmse_nmf = accuracy.rmse(preds_nmf, verbose=False)
            mae_nmf = accuracy.mae(preds_nmf, verbose=False)
            precision_nmf, recall_nmf = precision_recall_at_k(preds_nmf, k=k_nmf)
            st.session_state["nmf_metrics"] = {"RMSE": rmse_nmf, "MAE": mae_nmf, "Precision": precision_nmf, "Recall": recall_nmf}

        st.success("✅ Entraînement terminé et métriques calculées")

        # --- Affichage tables ---
        st.subheader("📌 Métriques SVD")
        st.table(st.session_state["svd_metrics"])
        st.subheader("📌 Métriques NMF")
        st.table(st.session_state["nmf_metrics"])

        # --- Graphiques Precision & Recall ---
        fig_prec, fig_rec = plot_precision_recall_separate(
            svd_prec=precision_svd, svd_rec=recall_svd,
            nmf_prec=precision_nmf, nmf_rec=recall_nmf
        )
        st.subheader("📈 Precision @k")
        st.pyplot(fig_prec)
        st.subheader("📈 Recall @k")
        st.pyplot(fig_rec)

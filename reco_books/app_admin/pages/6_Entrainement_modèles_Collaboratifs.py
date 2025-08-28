import streamlit as st
import pickle
from surprise import SVD, NMF, Dataset, Reader
from recommandation_de_livres.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_parquet

st.title("⚙️ Entraînement des modèles collaboratifs")

if 'DIR' not in st.session_state:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à la page d'accueil pour en choisir un.")
    st.stop()

# --- Choix du dataset ---
DIR = st.session_state.get("DIR", None)

if not DIR:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à l'accueil.")
    st.stop()

features_path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet"
if not features_path.exists():
    st.error(f"Fichier introuvable : {features_path}")
    st.stop()

collaborative_df = load_parquet(features_path)
st.info(f"📊 {len(collaborative_df)} notes chargées.")

# --- Onglets pour SVD et NMF ---
tab_svd, tab_nmf = st.tabs(["📈 SVD", "🔢 NMF"])

# =====================================================
# ====================== SVD ==========================
# =====================================================
with tab_svd:
    st.subheader("Paramètres SVD")

    # --- Hyperparamètres ---
    n_factors_svd = st.number_input("Nombre de facteurs latents", min_value=10, max_value=500, value=50, step=10, key="svd_n_factors")
    n_epochs_svd = st.number_input("Nombre d'epochs", min_value=5, max_value=200, value=50, step=5, key="svd_n_epochs")
    lr_all = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.002, step=0.001, format="%.4f", key="svd_lr")
    reg_all = st.number_input("Régularisation", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.4f", key="svd_reg")

    st.subheader("Échelle des notes")
    min_rating = st.number_input("Note minimale", min_value=1.0, max_value=10.0, value=1.0, step=0.5, key="svd_min_rating")
    max_rating = st.number_input("Note maximale", min_value=1.0, max_value=10.0, value=5.0, step=0.5, key="svd_max_rating")

    if min_rating >= max_rating:
        st.error("⚠️ La note minimale doit être inférieure à la note maximale.")
        st.stop()

    rating_scale_svd = (min_rating, max_rating)

    # --- Entraînement ---
    if st.button("🚀 Lancer l'entraînement SVD"):
        reader = Reader(rating_scale=rating_scale_svd)
        data = Dataset.load_from_df(collaborative_df[['user_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()

        svd = SVD(n_factors=n_factors_svd, n_epochs=n_epochs_svd, lr_all=lr_all, reg_all=reg_all)

        model_path = MODELS_DIR / DIR / "svd_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with st.spinner("Entraînement SVD en cours..."):
            svd.fit(trainset)

        with open(model_path, 'wb') as f:
            pickle.dump(svd, f)

        st.success(f"✅ Modèle SVD sauvegardé dans {model_path}")

# =====================================================
# ====================== NMF ==========================
# =====================================================
with tab_nmf:
    st.subheader("Paramètres NMF")

    # --- Hyperparamètres ---
    n_factors_nmf = st.number_input("Nombre de facteurs latents", min_value=10, max_value=500, value=50, step=10, key="nmf_n_factors")
    n_epochs_nmf = st.number_input("Nombre d'epochs", min_value=5, max_value=200, value=50, step=5, key="nmf_n_epochs")
    reg_pu = st.number_input("Régularisation utilisateurs", min_value=0.001, max_value=1.0, value=0.06, step=0.01, format="%.3f", key="nmf_reg_pu")
    reg_qi = st.number_input("Régularisation items", min_value=0.001, max_value=1.0, value=0.06, step=0.01, format="%.3f", key="nmf_reg_qi")

    st.subheader("Échelle des notes")
    min_rating_nmf = st.number_input("Note minimale", min_value=1.0, max_value=10.0, value=1.0, step=0.5, key="nmf_min_rating")
    max_rating_nmf = st.number_input("Note maximale", min_value=1.0, max_value=10.0, value=5.0, step=0.5, key="nmf_max_rating")

    if min_rating_nmf >= max_rating_nmf:
        st.error("⚠️ La note minimale doit être inférieure à la note maximale.")
        st.stop()

    rating_scale_nmf = (min_rating_nmf, max_rating_nmf)

    # --- Entraînement ---
    if st.button("🚀 Lancer l'entraînement NMF"):
        reader = Reader(rating_scale=rating_scale_nmf)
        data = Dataset.load_from_df(collaborative_df[['user_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()

        nmf = NMF(n_factors=n_factors_nmf, n_epochs=n_epochs_nmf, reg_pu=reg_pu, reg_qi=reg_qi)

        model_path = MODELS_DIR / DIR / "nmf_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with st.spinner("Entraînement NMF en cours..."):
            nmf.fit(trainset)

        with open(model_path, 'wb') as f:
            pickle.dump(nmf, f)

        st.success(f"✅ Modèle NMF sauvegardé dans {model_path}")

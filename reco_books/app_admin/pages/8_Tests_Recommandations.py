import streamlit as st
import pandas as pd
import numpy as np
import pickle
from surprise import SVD, Dataset, Reader
from pathlib import Path
from recommandation_de_livres.config import PROCESSED_DATA_DIR, MODELS_DIR
from recommandation_de_livres.loaders.load_data import load_parquet

# --- TITRE ---
st.title("🤖 Recommandations SVD + Diversité (MMR SBERT batch)")

# --- Choix du dataset ---
DIR = st.session_state.get("DIR", None)
if not DIR:
    st.error("⚠️ Aucun dataset sélectionné. Retournez à l'accueil.")
    st.stop()

# --- Chemins des fichiers ---
features_path = PROCESSED_DATA_DIR / DIR / "collaborative_dataset.parquet"
sbert_embeddings_path = PROCESSED_DATA_DIR / DIR / "embeddings_sbert.npy"
item_df_path = PROCESSED_DATA_DIR / DIR / "features_sbert.parquet"
content_path = PROCESSED_DATA_DIR / DIR / "content_dataset.parquet"
model_path = MODELS_DIR / DIR / "collaborative_model.pkl"

# --- Chargement des données ---
collab_df = load_parquet(features_path)
item_df = load_parquet(item_df_path)
content_df = load_parquet(content_path)
item_ids = item_df['item_id'].tolist()
sbert_embeddings = np.load(sbert_embeddings_path)

# --- Charger modèle SVD ---
with open(model_path, "rb") as f:
    svd_model = pickle.load(f)

# --- Sélection utilisateur ---
user_id = st.selectbox("Choisir un utilisateur", collab_df['user_id'].unique())

# --- Paramètres MMR ---
top_k = st.slider("Top K recommandations", 1, 50, 10)
lambda_param = st.slider("Poids de la pertinence (λ)", 0.0, 1.0, 0.7, 0.05)
top_candidates = st.slider("Nombre de candidats MMR", 10, 500, 100, 10)

# --- Fonction MMR batch ---
def mmr_batch(user_item_scores, item_embeddings, item_ids, top_k=10, lambda_param=0.7, top_candidates=100):
    iid_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    embeddings = np.array(item_embeddings)
    recommended_items_all = set()
    
    for uid, scores_dict in user_item_scores.items():
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_candidates]
        candidate_iids = [iid for iid, _ in sorted_items]
        candidate_scores = np.array([scores_dict[iid] for iid in candidate_iids])
        candidate_idx = [iid_to_idx[iid] for iid in candidate_iids]
        candidate_embeddings = embeddings[candidate_idx]

        selected = []
        unselected = np.arange(len(candidate_iids))

        for _ in range(min(top_k, len(candidate_iids))):
            if len(selected) == 0:
                best_idx = np.argmax(candidate_scores[unselected])
            else:
                sim_to_selected = np.max(
                    candidate_embeddings[unselected] @ candidate_embeddings[selected].T, axis=1
                )
                mmr_scores = lambda_param * candidate_scores[unselected] - (1 - lambda_param) * sim_to_selected
                best_idx = unselected[np.argmax(mmr_scores)]

            selected.append(best_idx)
            unselected = unselected[unselected != best_idx]

        recommended_items_all.update([candidate_iids[i] for i in selected])
    
    return recommended_items_all

# --- Bouton de recommandation ---
if st.button("Générer recommandations MMR batch"):
    # --- Générer scores SVD pour tous les utilisateurs ---
    user_ids = collab_df['user_id'].unique()
    user_item_scores = {}
    for uid in user_ids:
        user_items = set(collab_df[collab_df['user_id'] == uid]['item_id'])
        user_item_scores[uid] = {iid: svd_model.predict(uid, iid).est for iid in item_ids if iid not in user_items}

    # --- Calcul MMR batch ---
    recommended_items_all = mmr_batch(user_item_scores, sbert_embeddings, item_ids,
                                      top_k=top_k, lambda_param=lambda_param, top_candidates=top_candidates)

    # --- Catalog coverage ---
    catalog_coverage = len(recommended_items_all) / len(item_ids)
    st.success(f"📈 Catalog Coverage MMR batch : {catalog_coverage:.2%}")

    # --- Recommandations pour l'utilisateur sélectionné ---
    user_recs = mmr_batch({user_id: user_item_scores[user_id]}, sbert_embeddings, item_ids,
                          top_k=top_k, lambda_param=lambda_param, top_candidates=top_candidates)

    rec_df = content_df[content_df['item_id'].isin(user_recs)][['item_id', 'title', 'authors']]
    rec_df['ordre'] = range(1, len(rec_df)+1)
    st.subheader(f"Top {top_k} recommandations diversifiées pour {user_id}")
    st.dataframe(rec_df.sort_values('ordre').reset_index(drop=True))

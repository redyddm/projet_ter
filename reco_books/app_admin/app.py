import streamlit as st
import pandas as pd
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from recommandation_de_livres.iads.utils import choose_dataset_streamlit
# --------------------- INTERFACE ---------------------

st.title("📂 Sélection du dataset")

mode = st.radio("Type de données :", ["Raw", "Processed"], index=0)
datasets = choose_dataset_streamlit(raw=(mode == "Raw"))

if st.button("Charger"):
    st.info(f"Dataset {st.session_state['DIR']} choisi")
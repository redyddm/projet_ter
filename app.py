import streamlit as st
import pandas as pd
import numpy as np
import glob
import re
import os

@st.cache_data
def load_books():
    books = pd.read_pickle('./datasets/goodreads/perso/content_dataset_desc_en_final.pkl')
    return books

@st.cache_data
def load_ratings():
    ratings = pd.read_pickle('./datasets/goodreads/reco/collaborative_dataset_final.pkl')
    return ratings

books = load_books()
ratings = load_ratings()

st.title('Recommandation de livres')

if st.checkbox('Afficher les 20 premiers livres'):
    st.dataframe(books.head(20))

if st.checkbox('Afficher les 20 premieres notes'):
    st.dataframe(ratings.head(20))
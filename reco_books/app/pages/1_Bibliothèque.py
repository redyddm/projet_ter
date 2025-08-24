import streamlit as st
from pathlib import Path
import pandas as pd
from recommandation_de_livres.iads.utils import stars, stars_html

# ---- Répertoire des données ----
DIR = st.session_state['DIR']

books = st.session_state["books"]
ratings = st.session_state["ratings"]
users = st.session_state["users"]

# ---- Options de tri ----
sort_column = st.selectbox(
    "Trier par :",
    ["Titre", "Auteur", "Note moyenne"]
)

sort_order = st.selectbox(
    "Ordre :",
    ["Ascendant", "Descendant"]
)

# Mapping pour le DataFrame
sort_col_map = {
    "Titre": "title",
    "Auteur": "authors",
    "Note moyenne": "average_rating"
}

col_to_sort = sort_col_map[sort_column]
ascending = True if sort_order == "Ascendant" else False

# Préparer DataFrame pour tri
books_sorted = books.copy()
if col_to_sort == "average_rating":
    books_sorted[col_to_sort] = books_sorted[col_to_sort].fillna(0)
else:
    books_sorted[col_to_sort] = books_sorted[col_to_sort].fillna("")

books_sorted = books_sorted.sort_values(by=col_to_sort, ascending=ascending)

# ---- Pagination ----
if "page_num_bibl" not in st.session_state:
    st.session_state["page_num_bibl"] = 1

books_per_row = 5
books_per_page = 10
total_pages = (len(books_sorted) - 1) // books_per_page + 1

# Navigation
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅ Précédent") and st.session_state["page_num_bibl"] > 1:
        st.session_state["page_num_bibl"] -= 1
with col3:
    if st.button("Suivant ➡") and st.session_state["page_num_bibl"] < total_pages:
        st.session_state["page_num_bibl"] += 1
with col2:
    st.markdown(f"Page {st.session_state['page_num_bibl']} / {total_pages}")

# Indices de la page
start_idx = (st.session_state["page_num_bibl"] - 1) * books_per_page
end_idx = min(start_idx + books_per_page, len(books_sorted))
books_page = books_sorted.iloc[start_idx:end_idx]

# ---- Fonctions utilitaires ----
def safe_get(book, key, default="Inconnue"):
    val = book.get(key, default) if isinstance(book, dict) else book.get(key, default)
    if pd.isna(val):
        return default
    return val

def safe_year(book, key="year", default="Inconnue"):
    val = book.get(key, default) if isinstance(book, dict) else book.get(key, default)
    if pd.isna(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

# ---- Affichage ----
for row in range(0, len(books_page), books_per_row):
    cols = st.columns(books_per_row)
    for i, (_, book) in enumerate(books_page.iloc[row:row + books_per_row].iterrows()):
        col = cols[i % books_per_row]
        with col:
            st.image(safe_get(book, "image_url", "https://via.placeholder.com/150"), width=120)
            st.markdown(f"**{safe_get(book, 'title', 'Titre inconnu')}**")
            st.caption(safe_get(book, "authors", "Auteur inconnu"))
            if "average_rating" in book:
                st.markdown(stars_html(safe_get(book, "average_rating", 0)), unsafe_allow_html=True)

            with st.expander("📖 Voir détails"):
                st.write(f"**Auteur(s) :** {safe_get(book, 'authors')}")
                st.write(f"**Éditeur :** {safe_get(book, 'publisher')}")
                st.write(f"**Année :** {safe_year(book, 'year')}")
                st.write(f"**ISBN :** {safe_get(book, 'isbn', safe_get(book, 'item_id', 'N/A'))}")
                st.markdown("**Description :**")
                st.write(safe_get(book, "description", "Pas de description disponible."))

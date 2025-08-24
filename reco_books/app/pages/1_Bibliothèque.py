import streamlit as st
from pathlib import Path
from recommandation_de_livres.iads.utils import stars, stars_html

# ---- Répertoire des données ----
DIR = st.session_state['DIR']

books = st.session_state["books"].sort_values(by='title')
ratings = st.session_state["ratings"]
users = st.session_state["users"]

st.info(f"Dataset {DIR} chargé.")

# ---- Filtres et recherche ----
search_title = st.text_input("🔎 Rechercher par titre :")
authors_unique = sorted(books['authors'].dropna().unique())
categories_unique = sorted(
    set(cat.strip() for sublist in books['categories'].dropna().str.split(',') for cat in sublist)
)

author_filter = st.multiselect("Filtrer par auteur :", authors_unique)
category_filter = st.multiselect("Filtrer par catégorie :", categories_unique)

filtered_books = books.copy()

# Filtre par titre
if search_title:
    filtered_books = filtered_books[filtered_books['title'].str.contains(search_title, case=False, na=False)]

# Filtre par auteur
if author_filter:
    filtered_books = filtered_books[filtered_books['authors'].isin(author_filter)]

# Filtre par catégorie
if category_filter:
    filtered_books = filtered_books[filtered_books['categories'].apply(
        lambda x: any(cat.strip() in category_filter for cat in str(x).split(',')))
    ]

# ---- Pagination ----
if "page_num_bibl" not in st.session_state:
    st.session_state["page_num_bibl"] = 1

books_per_row = 5
books_per_page = 10
total_pages = (len(filtered_books) - 1) // books_per_page + 1

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
end_idx = min(start_idx + books_per_page, len(filtered_books))
books_page = filtered_books.iloc[start_idx:end_idx]

# ---- Affichage ----
import pandas as pd

def safe_get(book, key, default="Inconnue"):
    """Retourne la valeur du champ ou le défaut si None/NaN."""
    val = book.get(key, default) if isinstance(book, dict) else book.get(key, default)
    if pd.isna(val):
        return default
    return val

def safe_year(book, key="year", default="Inconnue"):
    """
    Retourne l'année sous forme d'entier si possible,
    sinon le défaut.
    """
    val = book.get(key, default) if isinstance(book, dict) else book.get(key, default)
    
    # Vérifie NaN ou None
    if pd.isna(val):
        return default
    
    # Essaie de convertir en int
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# Exemple d'intégration dans ta boucle
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

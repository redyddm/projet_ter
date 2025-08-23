import streamlit as st
from pathlib import Path
from recommandation_de_livres.iads.utils import stars
from recommandation_de_livres.loaders.load_data import load_csv

# ---- Répertoire des données ----
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Fonction de chargement ----
@st.cache_data
def load_books(path):
    return load_csv(path)

# ---- Titre de la page ----
st.title("🔍 Explorer tous les livres")

# ---- Chargement des données ----
if "books" not in st.session_state:
    books_path = DATA_DIR / "books_uniform.csv"  # à adapter selon ton dataset
    st.session_state["books"] = load_books(books_path)

books = st.session_state["books"]

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
for row in range(0, len(books_page), books_per_row):
    cols = st.columns(books_per_row)
    for i, (_, book) in enumerate(books_page.iloc[row:row + books_per_row].iterrows()):
        col = cols[i % books_per_row]
        with col:
            st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
            st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
            st.caption(book.get("authors", "Auteur inconnu"))
            if "average_rating" in book:
                st.markdown(stars(book.get("average_rating", 0)))

            with st.expander("📖 Voir détails"):
                st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                st.write(f"**Année :** {book.get('year', 'Inconnue')}")
                st.write(f"**ISBN :** {book.get('isbn') or book.get('item_id') or 'N/A'}")
                st.markdown("**Description :**")
                st.write(book.get("description", "Pas de description disponible."))

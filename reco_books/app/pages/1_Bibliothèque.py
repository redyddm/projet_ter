import streamlit as st
from pathlib import Path
from recommandation_de_livres.iads.utils import stars

from recommandation_de_livres.config import INTERIM_DATA_DIR
from recommandation_de_livres.loaders.load_data import load_csv

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_data
def load_books(path):
    return load_csv(path)

st.title("🔍 Explorer tous les livres")

# Vérification que les données sont chargées
choice = st.selectbox("Choix du dataset :", ["Recommender", "Goodreads", "Personnel"], index=0)

if "prev_dataset" not in st.session_state:
    st.session_state["prev_dataset"] = choice
    st.session_state["page_num_bibl"] = 1
elif st.session_state["prev_dataset"] != choice:
    st.session_state["page_num_bibl"] = 1
    st.session_state["prev_dataset"] = choice

if choice.startswith("Goodreads"):
        books = st.session_state["books_gdr"]
else:
    if choice.startswith("Personnel"):
        book_path = DATA_DIR / "books_uniform.csv"

    elif choice.startswith("Recommender"):
        DIR = "recommender" 
        book_path = INTERIM_DATA_DIR / DIR / "books_uniform.csv"

    books = load_books(book_path)

# Pagination
books_per_row = 5  # 5 livres par ligne
books_per_page = 10  # 10 livres par page
total_pages = (len(books) - 1) // books_per_page + 1

# Page courante
if "page_num_bibl" not in st.session_state:
    st.session_state["page_num_bibl"] = 1

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

# Calculer les indices pour la page actuelle
start_idx = (st.session_state["page_num_bibl"] - 1) * books_per_page
end_idx = start_idx + books_per_page
books_page = books.iloc[start_idx:end_idx]

# Affichage en lignes
for row in range(0, len(books_page), books_per_row):
    cols = st.columns(books_per_row)
    for i, (_, book) in enumerate(books_page.iloc[row:row+books_per_row].iterrows()):
        col = cols[i]
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


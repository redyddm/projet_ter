import streamlit as st
from recommandation_de_livres.iads.utils import stars

st.title("🔍 Explorer tous les livres")

# Vérification que les données sont chargées
if "books" not in st.session_state:
    st.error("❌ Les données ne sont pas chargées. Retournez à l’accueil.")
    st.stop()

books = st.session_state["books"]

# Pagination
books_per_page = 9  # nombre de livres par page
total_pages = (len(books) - 1) // books_per_page + 1

# Page courante
if "page_num" not in st.session_state:
    st.session_state["page_num"] = 1

# Navigation
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅ Précédent") and st.session_state["page_num"] > 1:
        st.session_state["page_num"] -= 1
with col3:
    if st.button("Suivant ➡") and st.session_state["page_num"] < total_pages:
        st.session_state["page_num"] += 1
with col2:
    st.markdown(f"Page {st.session_state['page_num']} / {total_pages}")

# Calculer les indices pour la page actuelle
start_idx = (st.session_state["page_num"] - 1) * books_per_page
end_idx = start_idx + books_per_page
books_page = books.iloc[start_idx:end_idx]

# Affichage en grille 3 colonnes
cols = st.columns(3)
for i, (_, book) in enumerate(books_page.iterrows()):
    col = cols[i % 3]
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
            st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")  
            st.markdown("**Description :**")
            st.write(book.get("description", "Pas de description disponible."))

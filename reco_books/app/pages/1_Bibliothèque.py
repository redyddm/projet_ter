import streamlit as st
from recommandation_de_livres.iads.app_ui import display_book_card

books = st.session_state["books"]
ratings = st.session_state["ratings"]

avg_ratings = ratings[["item_id", "average_rating"]].drop_duplicates()
books_avg = books.merge(avg_ratings, on="item_id", how="left")

# --- Tri ---
sort_column = st.selectbox("Trier par :", ["Titre", "Auteur", "Note moyenne"])
sort_order = st.selectbox("Ordre :", ["Croissant", "Décroissant"])

sort_col_map = {"Titre": "title", "Auteur": "authors", "Note moyenne": "average_rating"}
col_to_sort = sort_col_map[sort_column]
ascending = sort_order == "Croissant"

books_sorted = books_avg.copy()
books_sorted[col_to_sort] = books_sorted[col_to_sort].fillna("" if col_to_sort != "average_rating" else 0)
books_sorted = books_sorted.sort_values(by=col_to_sort, ascending=ascending)

# --- Recherche ---
search_query = st.text_input("🔎 Rechercher par titre :")
if search_query.strip():
    books_sorted = books_sorted[
        books_sorted["title"].str.contains(search_query, case=False, na=False)
    ]

# --- Pagination ---
if "page_num_bibl" not in st.session_state:
    st.session_state["page_num_bibl"] = 1
books_per_page = 10
total_pages = (len(books_sorted) - 1) // books_per_page + 1

start_idx = (st.session_state["page_num_bibl"] - 1) * books_per_page
end_idx = min(start_idx + books_per_page, len(books_sorted))
books_page = books_sorted.iloc[start_idx:end_idx]

# --- Affichage en grille ---
st.title("📚 Bibliothèque")
books_per_row = 5
for row in range(0, len(books_page), books_per_row):
    cols = st.columns(books_per_row)
    for i, (_, book) in enumerate(books_page.iloc[row:row+books_per_row].iterrows()):
        with cols[i % books_per_row]:
            display_book_card(book, allow_add=True, page_context="bibliotheque", show_rating_type="average")

# --- Navigation ---
col1, col2, col3 = st.columns([1,2,1])
with col1:
    if st.button("⬅ Précédent") and st.session_state["page_num_bibl"] > 1:
        st.session_state["page_num_bibl"] -= 1
        st.rerun()
with col3:
    if st.button("Suivant ➡") and st.session_state["page_num_bibl"] < total_pages:
        st.session_state["page_num_bibl"] += 1
        st.rerun()
with col2:
    st.markdown(f"Page {st.session_state['page_num_bibl']} / {total_pages}")

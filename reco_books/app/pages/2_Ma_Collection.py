import streamlit as st
from recommandation_de_livres.iads.app_ui import display_book_card
from recommandation_de_livres.iads.utils import imdb_weighted_rating
import pandas as pd

if not st.session_state.get("logged_in", False):
    st.warning("🚪 Veuillez vous connecter pour accéder à cette page.")
    st.stop()

books = st.session_state["books"]
ratings = st.session_state["ratings"]
user_index = st.session_state["user_index"]
user_id = st.session_state["user_id"]

if "ratings_count" not in books.columns:
    rating_count = ratings.groupby('item_id')['rating'].count().to_frame(name='ratings_count').reset_index()
    books = books.merge(rating_count, on='item_id', how='left')

    st.session_state["books"]=books


# --- Livres déjà notés par l'utilisateur ---
books_user = ratings[ratings["user_index"] == user_index]

# --- Merge dynamique pour récupérer seulement les colonnes absentes ---
cols_to_keep = [c for c in books.columns if c not in books_user.columns]
books_user_full = books_user.merge(
    books[cols_to_keep + ["item_id"]],
    on="item_id",
    how="left"
)

# --- Cold start : livres populaires pondérés IMDB ---
if books_user.empty:
    cold_start_books = imdb_weighted_rating(books).sort_values("weighted_rating", ascending=False).head(10)
    # Exclure les livres déjà notés (au cas où)
    cold_start_books = cold_start_books[~cold_start_books["item_id"].isin(books_user["item_id"])]
else:
    cold_start_books = pd.DataFrame() 

# --- Fusion pour afficher ensemble ---
books_to_display = pd.concat([books_user_full, cold_start_books], ignore_index=True)

st.title("📚 Ma Collection")
if books_to_display.empty:
    st.info("📭 Aucun livre trouvé pour cet utilisateur.")
else:
    # Pagination
    page_size = 5
    if "page_num_collection" not in st.session_state:
        st.session_state["page_num_collection"] = 0
    total_pages = (len(books_to_display)-1)//page_size + 1
    start_idx = st.session_state["page_num_collection"] * page_size
    end_idx = start_idx + page_size
    books_page = books_to_display.iloc[start_idx:end_idx]

    books_per_row = 5
    for row in range(0, len(books_page), books_per_row):
        cols = st.columns(books_per_row)
        for i, (_, book) in enumerate(books_page.iloc[row:row+books_per_row].iterrows()):
            with cols[i % books_per_row]:
                # Si le livre est déjà noté, pas de possibilité d'ajouter
                allow_add = book["item_id"] not in books_user["item_id"].values
                display_book_card(book, allow_add=allow_add, page_context="collection")

    # Navigation
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Précédent", key="prev_col") and st.session_state["page_num_collection"] > 0:
            st.session_state["page_num_collection"] -= 1
            st.rerun()
    with col3:
        if st.button("➡️ Suivant", key="next_col") and st.session_state["page_num_collection"] < total_pages-1:
            st.session_state["page_num_collection"] += 1
            st.rerun()
    with col2:
        st.caption(f"Page {st.session_state['page_num_collection']+1} / {total_pages}")

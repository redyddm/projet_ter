import streamlit as st
import pandas as pd
import numpy as np

from recommandation_de_livres.iads.app_ui import display_book_card, validate_pending_ratings
from recommandation_de_livres.iads.utils import imdb_weighted_rating

# --- Vérifier connexion ---
if not st.session_state.get("logged_in", False):
    st.warning("🚪 Veuillez vous connecter pour accéder à cette page.")
    st.stop()

books = st.session_state["books"]
ratings = st.session_state["ratings"]
user_index = st.session_state["user_index"]
user_id = st.session_state["user_id"]

# --- Initialiser panier temporaire si absent ---
if "pending_ratings" not in st.session_state:
    st.session_state["pending_ratings"] = []

# --- Livres déjà notés par l'utilisateur ---
books_user = ratings[ratings["user_index"] == user_index]

# --- Merge dynamique pour récupérer seulement les colonnes absentes ---
cols_to_keep = [c for c in books.columns if c not in books_user.columns]
books_user_full = books_user.merge(
    books[cols_to_keep + ["item_id"]],
    on="item_id",
    how="left"
)

st.title("📚 Ma Collection")

# --- Cas où utilisateur n'a pas de livres ---
if books_user_full.empty:
    st.info("📭 Aucun livre trouvé pour cet utilisateur.")
    # --- Suggestions populaires ---
    st.subheader("🔥 Suggestions populaires pour vous")

    popularity_df = ratings.groupby("item_id").agg(
        count=("rating", "count"),
        mean_rating=("rating", "mean")
    ).reset_index()

    m = popularity_df["count"].quantile(0.80)
    C = popularity_df["mean_rating"].mean()
    popularity_df["score"] = popularity_df.apply(
        lambda x: imdb_weighted_rating(x["count"], x["mean_rating"], m, C), axis=1
    )

    popularity_df = popularity_df.merge(books, on="item_id", how="inner")
    top_books = popularity_df.sort_values("score", ascending=False).head(100)
    recommended_books = top_books.head(5)

    cols = st.columns(5)
    for i, (_, book) in enumerate(recommended_books.iterrows()):
        with cols[i % 5]:
            display_book_card(book, allow_add=True, page_context="popular")

    # --- Affichage panier temporaire et bouton global ---
    if st.session_state.get("pending_ratings"):
        st.subheader("📝 Sélection en attente :")
        for r in st.session_state["pending_ratings"]:
            book_title = st.session_state["books"].set_index("item_id").loc[r["item_id"], "title"]
            st.write(f"- {book_title} ({r['rating']}⭐)")

        if st.button("Valider ma sélection"):
            validate_pending_ratings()


# --- Affichage de la collection ---
else:
    page_size = 5
    if "page_num_collection" not in st.session_state:
        st.session_state["page_num_collection"] = 0
    total_pages = (len(books_user_full)-1)//page_size + 1
    start_idx = st.session_state["page_num_collection"] * page_size
    end_idx = start_idx + page_size
    books_page = books_user_full.iloc[start_idx:end_idx]

    books_per_row = 5
    for row in range(0, len(books_page), books_per_row):
        cols = st.columns(books_per_row)
        for i, (_, book) in enumerate(books_page.iloc[row:row+books_per_row].iterrows()):
            with cols[i % books_per_row]:
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

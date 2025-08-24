import streamlit as st
import pandas as pd
from recommandation_de_livres.iads.utils import stars, save_df_to_parquet
from recommandation_de_livres.config import PROCESSED_DATA_DIR

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

def display_book_card(book, allow_add=True, page_context="reco", show_rating_type="user"):
    """
    show_rating_type : "user" pour rating personnel, "average" pour average_rating
    """
    user_index = st.session_state.get("user_index")
    user_id = st.session_state.get("user_id")
    ratings = st.session_state["ratings"]

    st.image(safe_get(book, "image_url", "https://via.placeholder.com/150"), width=120)
    st.markdown(f"**{safe_get(book, 'title')}**")
    st.caption(safe_get(book, "authors"))

    # Note à afficher
    if show_rating_type == "user":
        score_display = safe_get(book, "rating", 0)
    elif show_rating_type == "average":
        score_display = safe_get(book, "average_rating", 0)
    else:
        score_display = 0

    st.markdown(stars(score_display))

    # Bouton d'ajout (uniquement si allow_add et utilisateur connecté)
    if allow_add and user_index is not None and user_id is not None:
        already_in_collection = not ratings[
            (ratings["user_index"] == user_index) & 
            (ratings["item_id"] == book["item_id"])
        ].empty
        if already_in_collection:
            st.success("✅ Déjà dans votre collection")
        else:
            rating_key = f"user_{user_index}_book_{book['item_id']}_add_{page_context}"
            user_rating = st.slider("Votre note", 0, 5, 0, key=rating_key)
            if st.button("Ajouter à ma collection", key=f"add_{book['item_id']}_{page_context}"):
                new_entry = pd.DataFrame([{
                    "user_id": user_id,
                    "user_index": user_index,
                    "item_id": book["item_id"],
                    "rating": user_rating
                }])

                books=st.session_state['books']
                cols_to_keep = [c for c in books.columns if c not in new_entry.columns]
                books_user_full = new_entry.merge(
                    books[cols_to_keep + ["item_id"]],
                    on="item_id",
                    how="left"
)

                st.session_state["ratings"] = pd.concat([ratings, books_user_full], ignore_index=True)
                RATINGS_PATH = PROCESSED_DATA_DIR / st.session_state['DIR'] / "collaborative_dataset.parquet"
                save_df_to_parquet(st.session_state["ratings"], RATINGS_PATH)
                st.success(f"{book['title']} ajouté à votre collection avec {user_rating} ⭐")
                st.rerun()

    # Détails
    with st.expander("📖 Voir détails"):
        st.write(f"**Auteur(s) :** {safe_get(book, 'authors')}")
        st.write(f"**Éditeur :** {safe_get(book, 'publisher')}")
        st.write(f"**Année :** {safe_year(book, 'year')}")
        st.write(f"**ISBN :** {safe_get(book, 'isbn')}")
        st.markdown("**Description :**")
        st.write(safe_get(book, "description"))

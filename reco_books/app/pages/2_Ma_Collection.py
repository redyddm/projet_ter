import streamlit as st
import pandas as pd
from recommandation_de_livres.iads.utils import stars_html, stars_final
from recommandation_de_livres.iads.collabo_utils import rescale_ratings

if not st.session_state.get("logged_in", False):
    st.warning("🚪 Veuillez vous connecter pour accéder à cette page.")
    st.stop()

books = st.session_state["books"]
ratings = st.session_state["ratings"]
user_index = st.session_state['user_index']
user_id = st.session_state["user_id"]

# --- Section 1 : Ma Collection ---
st.title("📚 Ma Collection")

books_user = ratings[ratings["user_index"] == user_index].copy()

if books_user.empty:
    st.info("📭 Aucun livre trouvé pour cet utilisateur.")
else:
    st.subheader("Livres que vous avez déjà notés")
    page_size = 5
    total_pages = (len(books_user) - 1) // page_size + 1
    if "page_num" not in st.session_state:
        st.session_state["page_num"] = 0

    start_idx = st.session_state["page_num"] * page_size
    end_idx = start_idx + page_size
    books_page = books_user.iloc[start_idx:end_idx]

    cols = st.columns(5)
    for i, (_, book) in enumerate(books_page.iterrows()):
        col = cols[i % 5]
        with col:
            st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
            st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
            st.markdown(stars_html(book.get("rating", 0)), unsafe_allow_html=True)
            st.caption(book.get("authors", "Auteur inconnu"))

            with st.expander("📖 Voir détails"):
                st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                st.write(f"**Année :** {book.get('year', 'Inconnue')}")
                st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                st.markdown("**Description :**")
                st.write(book.get("description", "Pas de description disponible."))

    # Pagination
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Précédent", key="prev_top") and st.session_state["page_num"] > 0:
            st.session_state["page_num"] -= 1
    with col3:
        if st.button("➡️ Suivant", key="next_top") and st.session_state["page_num"] < total_pages - 1:
            st.session_state["page_num"] += 1
    with col2:
        st.caption(f"Page {st.session_state['page_num'] + 1} / {total_pages}")


# --- Section 2 : Ajouter des livres ---
st.title("➕ Ajouter des livres à ma collection")

user_books_ids = set(books_user['item_id'])
available_books = books[~books['item_id'].isin(user_books_ids)].copy()

search_input = st.text_input("Rechercher un livre par titre ou auteur :", key="search_add")
if search_input:
    mask = available_books['title'].str.contains(search_input, case=False, na=False) | \
           available_books['authors'].str.contains(search_input, case=False, na=False)
    available_books = available_books[mask]

page_size = 5
total_pages = (len(available_books) - 1) // page_size + 1
if "add_page_num" not in st.session_state:
    st.session_state["add_page_num"] = 0

start_idx = st.session_state["add_page_num"] * page_size
end_idx = start_idx + page_size
books_page = available_books.iloc[start_idx:end_idx]

cols = st.columns(5)
for i, (_, book) in enumerate(books_page.iterrows()):
    col = cols[i % 5]
    with col:
        st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
        st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
        st.caption(book.get("authors", "Auteur inconnu"))

         # Notation
        rating_key = f"user_{user_index}_book_{book['item_id']}_new"
        user_rating = st.slider(
            "Votre note", 0, 5, 0, key=rating_key
        )

        if st.button("Ajouter à ma collection", key=f"add_{book['item_id']}"):
            new_entry = pd.DataFrame([{
                "user_id": user_id,
                "user_index": user_index,
                "item_id": book["item_id"],
                "rating": user_rating
            }])
            st.session_state["ratings"] = pd.concat([st.session_state["ratings"], new_entry], ignore_index=True)
            st.success(f"{book['title']} ajouté à votre collection avec {user_rating} ⭐")


        with st.expander("📖 Voir détails"):
            st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
            st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
            st.write(f"**Année :** {book.get('year', 'Inconnue')}")
            st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
            st.markdown("**Description :**")
            st.write(book.get("description", "Pas de description disponible."))

# Pagination des livres à ajouter
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ Précédent (Ajouter)", key="prev_add") and st.session_state["add_page_num"] > 0:
        st.session_state["add_page_num"] -= 1
with col3:
    if st.button("➡️ Suivant (Ajouter)", key="next_add") and st.session_state["add_page_num"] < total_pages - 1:
        st.session_state["add_page_num"] += 1
with col2:
    st.caption(f"Page {st.session_state['add_page_num'] + 1} / {total_pages}")

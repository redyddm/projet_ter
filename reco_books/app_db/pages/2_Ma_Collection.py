import streamlit as st
from recommandation_de_livres.iads.utils import stars


if not st.session_state.get("logged_in", False):
    st.warning("🚪 Veuillez vous connecter pour accéder à cette page.")
    st.stop()

st.title("📚 Ma Collection")

# Vérifier que les données sont chargées
if "books" not in st.session_state or "ratings" not in st.session_state:
    st.error("❌ Les données ne sont pas chargées. Retournez à l’accueil.")
    st.stop()

books = st.session_state["books"]
ratings = st.session_state["ratings"]
users = st.session_state["users"]    

user_index = st.session_state['user_index']
user_id = st.session_state["user_id"]
username = st.session_state["username"]


# --- Sidebar connexion ---
if st.session_state.get("logged_in", False):
    st.sidebar.write(f"✅ Connecté : user {st.session_state['user_index']}")
    if st.sidebar.button("Se déconnecter", key="logout"):
        st.session_state["logged_in"] = False
        st.session_state["user_index"] = None
        st.session_state["user_id"] = None
        if "page_num" in st.session_state:
            del st.session_state["page_num"]

# --- Affichage bibliothèque ---
books_user = ratings[ratings["user_index"] == user_index]

if books_user.empty:
    st.info("📭 Aucun livre trouvé pour cet utilisateur.")
else:
    # Pagination
    page_size = 10
    total_books = len(books_user)
    total_pages = (total_books - 1) // page_size + 1

    # Initialiser la page si elle n'existe pas
    if "page_num" not in st.session_state:
        st.session_state["page_num"] = 0

    # Boutons de navigation en haut
    if "prev_clicked" not in st.session_state:
        st.session_state.prev_clicked = False
    if "next_clicked" not in st.session_state:
        st.session_state.next_clicked = False

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Précédent", key="prev_top"):
            st.session_state.prev_clicked = True
    with col3:
        if st.button("➡️ Suivant", key="next_top"):
            st.session_state.next_clicked = True

    # Mettre à jour la page
    if st.session_state.prev_clicked and st.session_state["page_num"] > 0:
        st.session_state["page_num"] -= 1
        st.session_state.prev_clicked = False
    if st.session_state.next_clicked and st.session_state["page_num"] < total_pages - 1:
        st.session_state["page_num"] += 1
        st.session_state.next_clicked = False

    start_idx = st.session_state["page_num"] * page_size
    end_idx = start_idx + page_size
    books_page = books_user.iloc[start_idx:end_idx]

    # Affichage en grille 5 colonnes
    cols = st.columns(5)
    for i, (_, book) in enumerate(books_page.iterrows()):
        col = cols[i % 5]
        with col:
            st.image(book.get("image_url", "https://via.placeholder.com/150"), width=120)
            st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
            st.markdown(stars(book.get("rating", 0)))
            st.caption(book.get("authors", "Auteur inconnu"))

            with st.expander("📖 Voir détails"):
                st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}") 
                st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                st.write(f"**Année :** {book.get('year', 'Inconnue')}")
                st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                st.markdown("**Description :**")
                st.write(book.get("description", "Pas de description disponible."))

    # Boutons de navigation en bas (mêmes clés que pour le haut)
    #col1, col2, col3 = st.columns([1, 2, 1])
    #with col1:
    #    if st.button("⬅️ Précédent", key="prev_bottom"):
    #        st.session_state.prev_clicked = True
    #with col3:
    #    if st.button("➡️ Suivant", key="next_bottom"):
    #        st.session_state.next_clicked = True

    # Afficher le numéro de page
    st.caption(f"Page {st.session_state['page_num'] + 1} / {total_pages}")

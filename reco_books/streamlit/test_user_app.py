import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from recommandation_de_livres.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from recommandation_de_livres.iads.utils import stars

DIR = 'goodreads'

user_ratings = pd.read_csv(PROCESSED_DATA_DIR / DIR / "collaborative_dataset.csv")
books = pd.read_csv(INTERIM_DATA_DIR / DIR / "books_authors.csv")


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None

if not st.session_state["logged_in"]:
    # Sélecteur d'utilisateur (plus simple que le number_input)
    user_id_input = st.selectbox(
        "Sélectionnez votre user_id",
        sorted(user_ratings["user_index"].unique())
    )

    if st.button("Se connecter"):
        if user_id_input in user_ratings["user_index"].values:
            st.session_state["logged_in"] = True
            st.session_state["user_id"] = user_id_input
        else:
            st.error("Utilisateur non trouvé")

if st.session_state["logged_in"]:
    st.sidebar.write(f"✅ Connecté : user {st.session_state['user_id']}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state["logged_in"] = False
        st.session_state["user_id"] = None

    # Filtrer les livres de l'utilisateur connecté
    books_user = user_ratings[user_ratings["user_index"] == st.session_state["user_id"]]

    if not books_user.empty:
        st.subheader("📚 Ma Bibliothèque")

        # Grille d'affichage (3 colonnes)
        cols = st.columns(3)
        for i, (_, book) in enumerate(books_user.iterrows()):
            col = cols[i % 3]
            with col:
                st.image(
                    book.get("image_url", "https://via.placeholder.com/150"),
                    width=120
                )
                st.markdown(f"**{book.get('title', 'Titre inconnu')}**")
                st.markdown(stars(book.get("rating", 0)))
                st.caption(book.get("authors", "Auteur inconnu"))

                # Détails en expander
                with st.expander("📖 Voir détails"):
                    st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                    st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                    st.write(f"**Année :** {book.get('year', 'Inconnue')}")
                    st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                    #st.write(f"**Note donnée :** {book.get('rating', 'Non noté')}")
                    st.markdown("**Description :**")
                    st.write(book.get("description", "Pas de description disponible."))
    else:
        st.subheader("📚 Ma Bibliothèque")
        st.write("Aucun livre trouvé pour cet utilisateur")
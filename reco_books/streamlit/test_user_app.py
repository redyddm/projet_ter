import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from recommandation_de_livres.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from recommandation_de_livres.preprocessing.preprocess_user_book import rename_columns_users


DIR = 'Recommender_dataset'

user_ratings = pd.read_csv(PROCESSED_DATA_DIR / "user_book_dataset.csv")
users = pd.read_csv(RAW_DATA_DIR / DIR / "Users.csv")
users =  rename_columns_users(users)


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None

if not st.session_state["logged_in"]:
    user_id_input = st.number_input("Entrez votre user_id", min_value=int(users['user_id'].min()), 
                                    max_value=int(users['user_id'].max()), step=1)

    if st.button("Se connecter"):
        if user_id_input in users['user_id']:
            st.session_state["logged_in"] = True
            st.session_state["user_id"] = user_id_input
        else:
            st.error("Utilisateur non trouvé")

if st.session_state["logged_in"]:
    st.sidebar.write(f"Connecté : user {st.session_state['user_id']}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state["logged_in"] = False
        st.session_state["user_id"] = None

    # Filtrer les livres de l'utilisateur connecté
    books = user_ratings[user_ratings["user_id"] == st.session_state["user_id"]].to_dict("records")

    if books:
        st.subheader("📚 Ma Bibliothèque")

        # Grille d'affichage (3 colonnes)
        cols = st.columns(3)
        for i, book in enumerate(books):
            col = cols[i % 3]
            with col:
                st.image(
                    book.get("Image-URL-L", "https://via.placeholder.com/150"),
                    width=120
                )
                st.markdown(f"**{book['title']}**")
                st.caption(book['authors'])

                # Détails en expander
                with st.expander("📖 Voir détails"):
                    #st.image(book.get("Image-URL-L", "https://via.placeholder.com/150"), width=150)
                    st.write(f"**Auteur(s) :** {book.get('authors', 'Inconnu')}")
                    st.write(f"**Éditeur :** {book.get('publisher', 'Inconnu')}")
                    st.write(f"**Année :** {book.get('year', 'Inconnue')}")
                    st.write(f"**ISBN :** {book.get('isbn', 'N/A')}")
                    st.markdown("**Description :**")
                    st.write(book.get("description", "Pas de description disponible."))

    else:
        st.subheader("📚 Ma Bibliothèque")
        st.write("Aucun livre dans la bibliothèque")


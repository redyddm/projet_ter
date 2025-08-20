import streamlit as st
import pandas as pd

# -------------------------
# Utilisateurs et mots de passe
# -------------------------
users = {
    "user1": "password1",
    "user2": "password2",
    "user3": "password3"
}

# -------------------------
# Bibliothèques des utilisateurs
# -------------------------
user_books = {
    "user1": [
        {"title": "Harry Potter and the Chamber of Secrets", "author": "J.K. Rowling", "cover_url": "https://covers.openlibrary.org/b/isbn/0439064872-L.jpg", "description": "Harry returns for his second year..."},
        {"title": "The Hobbit", "author": "J.R.R. Tolkien", "cover_url": "https://covers.openlibrary.org/b/isbn/0345339681-L.jpg", "description": "Bilbo sets out on an unexpected journey..."},
    ],
    "user2": [
        {"title": "The Little Prince", "author": "Antoine de Saint-Exupéry", "cover_url": "https://covers.openlibrary.org/b/isbn/0156012197-L.jpg", "description": "A young prince explores the universe..."},
    ],
    "user3": []
}

# -------------------------
# Session login
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None

# -------------------------
# Formulaire de connexion
# -------------------------
if not st.session_state["logged_in"]:
    st.subheader("🔐 Connexion")
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")
        if submitted:
            if username in users and users[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Connecté en tant que {username}")
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")

# -------------------------
# Interface utilisateur connectée
# -------------------------
if st.session_state["logged_in"]:
    st.sidebar.write(f"Connecté : {st.session_state['username']}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None

    books = user_books.get(st.session_state["username"], [])
    if books:
        st.subheader("📚 Ma Bibliothèque")
        cols = st.columns(3)  # 3 livres par ligne
        for idx, book in enumerate(books):
            col = cols[idx % 3]
            col.image(book["cover_url"], width=150)
            col.markdown(f"**{book['title']}**")
            col.caption(f"✍️ {book['author']}")
            if col.button("Voir détails", key=f"{book['title']}_{idx}"):
                st.markdown("---")
                st.subheader(book["title"])
                st.caption(f"✍️ {book['author']}")
                st.image(book["cover_url"], width=200)
                st.write(book["description"])
    else:
        st.info("Votre bibliothèque est vide.")

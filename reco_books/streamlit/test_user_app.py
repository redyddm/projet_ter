import streamlit as st
import pandas as pd

# Exemple de dataset (dans ton cas tu chargeras depuis ton pickle)
books = pd.DataFrame([
    {"title": "Harry Potter and the Chamber of Secrets", 
     "author": "J.K. Rowling", 
     "cover_url": "https://covers.openlibrary.org/b/isbn/0439064872-L.jpg",
     "description": "Harry Potter's second year at Hogwarts..."},
    {"title": "The Hobbit", 
     "author": "J.R.R. Tolkien", 
     "cover_url": "https://covers.openlibrary.org/b/isbn/0345339681-L.jpg",
     "description": "Bilbo Baggins goes on an adventure..."},
    {"title": "The Little Prince", 
     "author": "Antoine de Saint-Exupéry", 
     "cover_url": "https://covers.openlibrary.org/b/isbn/0156012197-L.jpg",
     "description": "A story about a prince from another planet..."},
])

st.title("📚 Ma Bibliothèque")

# Nombre de colonnes (3 livres par ligne par ex.)
cols_per_row = 3

# Itérer sur les livres par paquets
for i in range(0, len(books), cols_per_row):
    cols = st.columns(cols_per_row)
    for col, (_, book) in zip(cols, books.iloc[i:i+cols_per_row].iterrows()):
        with col:
            st.image(book["cover_url"], width=150)
            st.write(f"**{book['title']}**")
            st.caption(f"✍️ {book['author']}")
            
            # Bouton voir détails
            if st.button("Voir détails", key=book["title"]):
                st.session_state["selected_book"] = book.to_dict()

# Si un livre a été cliqué
if "selected_book" in st.session_state:
    book = st.session_state["selected_book"]
    st.markdown("---")
    st.subheader(book["title"])
    st.caption(f"✍️ {book['author']}")
    st.image(book["cover_url"], width=200)
    st.write(book["description"])

# 

## Project Organization

```
- .env
- .gitignore
- **app/**
  - Accueil.py
  - **pages/**
    - 1_Bibliothèque.py
    - 2_Ma_Collection.py
    - 3_Recommandations.py
- **app_admin/**
  - Accueil.py
  - **data/**
    - **raw/**
      - books_user.parquet
      - ratings_user.parquet
  - **pages/**
    - ... (8 items)
- **data/**
  - **external/**
    - .gitkeep
  - **interim/**
    - .gitkeep
    - **goodreads/**
      - ... (17 items)
    - **recommender/**
      - ... (20 items)
  - **processed/**
    - .gitkeep
    - **goodreads/**
      - ... (19 items)
    - **recommender/**
      - ... (23 items)
  - **raw/**
    - .gitkeep
    - **goodreads/**
      - ... (6 items)
    - **recommender/**
      - Books.csv
      - Ratings.csv
      - Users.csv
- **docs/**
  - .gitkeep
  - **docs/**
    - getting-started.md
    - index.md
  - mkdocs.yml
  - README.md
- Makefile
- **models/**
  - .gitkeep
  - **goodreads/**
    - ... (11 items)
  - **recommender/**
    - ... (11 items)
- **notebooks/**
  - .gitkeep
  - notebook.ipynb
- **open_library/**
  - openlibrary_extract.py
- pyproject.toml
- **recommandation_de_livres/**
  - ... (9 items)
- redame.py
- **references/**
  - .gitkeep
- **reports/**
  - .gitkeep
  - **figures/**
    - .gitkeep
    - precision_at_k.png
    - recall_at_k.png
- requirements.txt
```

### Notes

- Les dossiers volumineux ou contenant de nombreux fichiers sont résumés.
- Fichiers temporaires ou caches sont ignorés.

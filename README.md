# Recommandation de livres

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Un système de recommandation de livres

## Project Organization

```
notebooks
├── datasets                <- dossier avec différents datasets
├── package                 <- dossier avec différentes fonctions utilisées
├── param                   <- dossier avec les meilleurs paramètres de SVD et NMF
├── goodreads.ipynb         <- notebook avec dataset goodreads [text](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)
└── reco_depository.ipynb   <- notebook avec dataset depository ([text](https://www.kaggle.com/datasets/sp1thas/book-depository-dataset)) 


datasets                    <- dossier avec dataset initial ([text](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset))


reco_books                  <- dossier principal du projet 
├── .env
├── .gitignore
├── Makefile
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.cfg
|
|
├── app                     <- dossier de l'interface streamlit utilisateur
│ ├── pages                 <- dossier des pages de l'interface
│ └── Accueil.py            <- page d'accueil de l'interface
│
├── app_admin               <- dossier de l'interface streamlit administrateur  
│ ├── pages                 <- dossier des pages de l'interface
│ └── Accueil.py            <- page d'accueil de l'interface
│
├── data                    <- données du projet
│ ├── external
│ ├── interim               <- données intermédiaires (quelques prétraitements)
│ ├── processed             <- données prêtes 
│ │ ├── ... (nombreux fichiers)
│ └── raw                   <- données brutes
│ ├── ... (nombreux fichiers)
│
├── docs
│ ├── docs
│ └── mkdocs.yml
│
├── models                  <- dossier des modèles
├── notebooks               <- dossier des notebooks
├── open_library            <- dossier contenant les scripts pour récupérer les informations supplémentaires de livres│
└── recommandation_de_livres <- module contenant les étapes principales
  ├── init.py
  ├── config.py             <- fichier avec différentes variables globales
  ├── build_dataset         <- dossier avec les scripts créant les datasets
  ├── dataset               <- dossier avec les scripts appelant build_dataset pour créer les versions finales des datasets
  ├── features              <- dossier avec les scripts permettant de créer les features
  ├── modeling              <- dossier avec les scripts permettant d'entrainer et tester les modèles
  │ ├── predict
  │ └── train
  ├── iads                  <- dossier avec les scripts de fonctions usuelles
  ├── loaders               <- dossier avec script chargeant les données
  └── preprocessing         <- dossier avec les scripts permettant de faire le prétraitement des données


```

### Notes

- Les dossiers volumineux ou contenant de nombreux fichiers sont résumés avec `... (nombreux fichiers)`.  
- Les fichiers temporaires ou caches (`__pycache__`, `.pyc`, `.git`) sont ignorés.  
- Ce README donne une **vue d’ensemble** de la structure du projet.

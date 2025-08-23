import pandas as pd
from recommandation_de_livres.loaders.load_data import load_parquet

def create_users_file(ratings_csv_path, output_csv_path="users.csv"):
    """
    Génère un fichier users.csv avec les colonnes:
    user_id, user_index, username
    basé sur les user_id présents dans le fichier de ratings.
    
    Args:
        ratings_csv_path (str): Chemin vers le CSV contenant les ratings avec colonne 'user_id'.
        output_csv_path (str): Chemin de sortie pour le CSV des utilisateurs.
    """
    # Charger uniquement la colonne user_id pour économiser de la mémoire
    ratings = pd.read_csv(ratings_csv_path, usecols=['user_id'])
    
    # Extraire les user_id uniques
    unique_user_ids = ratings['user_id'].drop_duplicates().tolist()
    
    # Créer un mapping user_id -> user_index
    user_mapping = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    
    # Construire le DataFrame final
    users_df = pd.DataFrame({
        'user_id': list(user_mapping.keys()),
        'user_index': list(user_mapping.values()),
    })
    users_df['username'] = 'user' + users_df['user_index'].astype(str)
    
    # Sauvegarder
    users_df.to_csv(output_csv_path, index=False)
    print(f"Fichier {output_csv_path} généré avec {len(users_df)} utilisateurs.")


import pandas as pd

def create_users_file(ratings_csv_path, output_csv_path="users.csv"):
    """
    Génère un fichier users.csv avec les colonnes:
    user_id, user_index, username
    basé sur les user_index présents dans le fichier de ratings.
    
    Args:
        ratings_csv_path (str): Chemin vers le CSV contenant les ratings avec colonnes 'user_id' et 'user_index'.
        output_csv_path (str): Chemin de sortie pour le CSV des utilisateurs.
    """
    # Charger uniquement les colonnes nécessaires
    ratings = pd.read_csv(ratings_csv_path, usecols=['user_id', 'user_index'])
    
    # Extraire les combinaisons uniques user_id / user_index
    users_df = ratings.drop_duplicates(subset=['user_id', 'user_index']).copy()
    
    # Ajouter le username basé sur user_index
    users_df['username'] = 'user' + users_df['user_index'].astype(str)
    
    # Sauvegarder
    users_df.to_csv(output_csv_path, index=False)
    print(f"Fichier {output_csv_path} généré avec {len(users_df)} utilisateurs.")

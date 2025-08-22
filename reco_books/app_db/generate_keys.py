import pickle
from pathlib import Path
import pandas as pd

import streamlit_authenticator as stauth

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommandation_de_livres.config import DB_PARAMS

def get_engine():
    from sqlalchemy import create_engine
    return create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}/{DB_PARAMS['dbname']}"
    )

engine = get_engine()

def load_users():
    query = "SELECT * FROM users"
    return pd.read_sql(query, engine)

users = load_users()
users.sort_values(by='user_index', inplace=True)

print(users[:2])

passwords = users['username']

hashed_pwd=stauth.Hasher.hash_list(passwords)

file_path = Path(__file__).parent / "hashed_pwd.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_pwd, file)



import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth



# ----- USER AUTHENTIFICATOR --------

users = pd.read_csv("D:\TER\reco_books\data\raw\goodreads\users.csv")
names = users['user_index']
usernames = users['username']

file_path = Path(__file__).parent / "hashed_pwd.pkl"
with file_path.open("rb") as file:
    hashed_pwd = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_pwd, "reco_books", "userrandom", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:

    authenticator.logout("Logout", "sidebar")

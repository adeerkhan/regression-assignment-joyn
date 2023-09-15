import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor

st.title("Regression Model")

st.write("""
# Explore different Regression Models
Which one is the best?
""")

# For creating a selection box with options
uploaded_file = st.file_uploader("Upload a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_file:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    raw_data = pd.read_csv(uploaded_file.name)
    st.write(raw_data.head())

classifier_name = st.sidebar.selectbox("Select Regression Model",
                                    ("LinearRegression", "RandomForest", "XGBRFRegressor"))

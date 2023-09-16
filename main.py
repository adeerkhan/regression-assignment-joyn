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
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRFRegressor



# one-hot-encoding data function
#def adding_more_columns(data):
    

  #  return new_data

@st.cache_data
def one_hot_encode_data(data, columns_to_encode):
    # Use pd.get_dummies to one-hot encode specified columns
    encoded_def = pd.get_dummies(data, columns=columns_to_encode, drop_first=True)
    return encoded_def

def main():
    
    st.title("Regression Model")
    st.write("""
    # Explore different Regression Models
    Which one gives the best result for your data?
    """)
    # for creating a selection box with options
    uploaded_file = st.file_uploader("Upload a CSV file", accept_multiple_files=True)
    for uploaded_file in uploaded_file:
        st.write("Data of file:", uploaded_file.name)
        raw_data = pd.read_csv(uploaded_file.name)
        st.write(raw_data.head())

    # get a list of categorical columns from the user
    st.sidebar.header("Select categorical columns for One-Hot Encoding")
    categorical_columns = st.sidebar.multiselect("Select columns", raw_data.columns)

    if st.sidebar.button("Run One-Hot Encode"):
        encoded_df = one_hot_encode_data(raw_data, categorical_columns)
        st.subheader("All data in numericals now")
        st.write(encoded_df.head())
        st.sidebar.header("Want to find insights from your CSV?")
        if encoded_df is not None:
            if st.sidebar.button("Generate Describe Table"):
                st.subheader("Describe Table")
                st.write(encoded_df.describe())
            if st.sidebar.button("Generate Info Table"):
                st.subheader("Info Table")
                st.write(encoded_df.info())
            if st.sidebar.button("Generate IsNull Table"):
                st.subheader("")
                st.write(encoded_df.isnull().sum())
        else:
            st.sidebar.warning("No data to take insight from. Please click 'Run One-Hot Encode' to generate encoded data.")

    # get a list of categorical columns from the user
    st.subheader("Select columns you want to join to reduce dimensionality")
    categorical_columns = st.multiselect("Select columns", encoded_df.columns)

    classifier_name = st.sidebar.selectbox("Select Regression Model",
                                    ("LinearRegression", "RandomForest", "XGBRFRegressor"))

def describe_info(describe):
    if describe is not None:
        if st.sidebar.button("Generate Describe Table"):
            st.subheader("Describe Table")
            st.write(describe.describe())
        if st.sidebar.button("Generate Info Table"):
            st.subheader("Info Table")
            st.write(describe.info())
        if st.sidebar.button("Generate IsNull Table"):
            st.subheader("")
            st.write(describe.isnull().sum())
    else:
        st.sidebar.warning("No data to take insight from. Please click 'Run One-Hot Encode' to generate encoded data.")

if __name__ == '__main__':
    main()


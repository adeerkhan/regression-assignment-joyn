import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRFRegressor




# one-hot encoding function
def one_hot_encode_data(data, columns_to_encode):
    # Use pd.get_dummies to one-hot encode specified columns
    encoded_def = pd.get_dummies(data, columns=columns_to_encode, drop_first=True)
    return encoded_def

# columns join function
def columns_join(data, name, columns_to_join):
    data[name] = data[columns_to_join].sum(axis=1)
    data.drop(columns=columns_to_join, inplace=True)
    return data

def main():
    global encoded_df
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
        st.table(raw_data.head())
        
        # get a list of categorical columns from the user to encode
        st.sidebar.header("Select categorical columns for One-Hot Encoding")
        categorical_columns = st.sidebar.multiselect("Select columns", raw_data.columns)
        encoded_df = one_hot_encode_data(raw_data, categorical_columns)

        # encode button
        if st.sidebar.button("Run One-Hot Encode"):
            st.subheader("All data in numericals now")
            st.table(encoded_df.head())
            st.sidebar.header("Want to find insights from your CSV?")

        # generating description table
        if st.sidebar.button("Generate Describe Table"):
            st.subheader("Description Table")
            st.table(encoded_df.describe())

        # generating information table
        if st.sidebar.button("Generate Info Table"):
            st.subheader("Information Table")
            info_buffer = io.StringIO()
            encoded_df.info(buf=info_buffer)
            info_text = info_buffer.getvalue()
            st.text(info_text)

        # generating isnull table
        if st.sidebar.button("Generate IsNull Table"):
            null_data = encoded_df.isnull().sum().sum()
            
            if null_data == 0:
                st.subheader("Congrats there are no Nulls in your data")
            else:
                st.subheader("Fix the Nulls in your Data")
            st.write("Total null values:", null_data)
            st.subheader("IsNull Table")         
            st.table(encoded_df.isnull().sum())
            
        # get a list of categorical columns from the user that he wants to join
        st.subheader("Select columns you want to join to reduce dimensionality")
        join_columns = st.multiselect("Select columns", encoded_df.columns)
        name = title = st.text_input('Enter the name of your new column in which you want to add your selected column data and  then press "Enter"', ' ')
        if st.sidebar.button("Join Columns"):
            st.subheader("New Columns")
            new_df = columns_join(encoded_df,name, join_columns)
            st.table(encoded_df.head())

        classifier_name = st.sidebar.selectbox("Select Regression Model",
                                        ("LinearRegression", "RandomForest", "XGBRFRegressor"))



if __name__ == '__main__':
    main()


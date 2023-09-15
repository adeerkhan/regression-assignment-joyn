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

encoded_data = None

# one-hot-encoding data function
def one_hot_encode_data(data, columns_to_encode):
    encoder = OneHotEncoder(sparse=False, drop=None)
    encode_data = encoder.fit_transform(data[columns_to_encode])
    encoded_df = pd.DataFrame(encode_data, columns=encoder.get_feature_names_out(input_features=columns_to_encode))
    return encoded_df

st.title("Regression Model")

st.write("""
# Explore different Regression Models
Which one gives the best result for your data?
""")

# for creating a selection box with options
uploaded_file = st.file_uploader("Upload a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_file:
    bytes_data = uploaded_file.read()
    st.write("Data of file:", uploaded_file.name)
    raw_data = pd.read_csv(uploaded_file.name)
    st.write(raw_data.head())

 # get a list of categorical columns from the user
st.sidebar.header("Select categorical columns for One-Hot Encoding")
categorical_columns = st.sidebar.multiselect("Select columns", raw_data.columns)

if st.sidebar.button("Run One-Hot Encode"):
    if len(categorical_columns) > 0:
        encoded_df = one_hot_encode_data(raw_data, categorical_columns)
        other_columns = raw_data.select_dtypes(include=['int64', 'float64']).columns
        encoded_data = pd.concat([raw_data[other_columns], encoded_df], axis=1)
        st.subheader("All data in numericals now")
        st.write(encoded_data.head())
    else:
        st.warning("Please select at least one categorical column for encoding.")


if encoded_data is not None:
    st.sidebar.header("Want to find insights from your CSV?")
    if st.sidebar.button("Generate Describe Table"):
        st.subheader("Describe Table")
        st.write(encoded_data.describe())
    if st.sidebar.button("Generate Info Table"):
        st.subheader("Info Table")
        st.write(encoded_data.info())
    if st.sidebar.button("Generate IsNull Table"):
        st.subheader("")
        st.write(encoded_data.isnull().sum())
else:
    st.sidebar.warning("No data to describe. Please click 'Run One-Hot Encode' to generate encoded data.")



classifier_name = st.sidebar.selectbox("Select Regression Model",
                                    ("LinearRegression", "RandomForest", "XGBRFRegressor"))



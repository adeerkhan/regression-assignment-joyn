import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor

# calculating metrics to evaluate the model 
def find_metrics(y_test, y_pred):
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    abc = 'Root Mean Squared Error (RMSE): ' + str(rmse)
    st.write(abc)
    # calculate R2 Score
    r2 = r2_score(y_test, y_pred)
    abd = 'R2 Score: ' + str(r2)
    st.write(abd)


# graph for plotting the results for regression predictions
def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots()
    plt.scatter(y_test, y_pred, alpha=0.5, label='Scatter Plot')

    # add a line representing a perfect fit
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    st.pyplot(fig)

# one-hot encoding function
def one_hot_encode_data(data, columns_to_encode):
    # use pd.get_dummies to one-hot encode specified columns
    encoded_def = pd.get_dummies(data, columns=columns_to_encode, drop_first=True)
    return encoded_def

# columns join function
def columns_join(data, name, columns_to_join):
    data[name] = data[columns_to_join].sum(axis=1)
    data.drop(columns=columns_to_join, inplace=True)
    return data

def main():
    global encoded_df
    st.title("Regression Models")
    st.write("""
    # Explore different Regression Models
    Which one gives the best result for your data?
    """)

    # for creating a selection box with options
    uploaded_file = st.file_uploader("Upload a CSV file here", accept_multiple_files=True)
    for uploaded_file in uploaded_file:
        st.write("Data of file:", uploaded_file.name)
        raw_data = pd.read_csv(uploaded_file.name)
        st.table(raw_data.head())
        
        # get a list of categorical columns from the user to encode
        st.sidebar.header("One-Hot Encoding")
        categorical_columns = st.sidebar.multiselect("Select columns", raw_data.columns)
        encoded_df = one_hot_encode_data(raw_data, categorical_columns)
        # encode button
        if st.sidebar.button("Run One-Hot Encode"):
            
            st.subheader("All data in numericals now")
            st.table(encoded_df.head())
            
        st.sidebar.header("Want to find insights from your CSV?")
        st.session_state.insight_name = st.sidebar.selectbox("Select Table to View",
                            ("","Describe Table", "Info Table", "IsNull Table"))
        # generating description table
        if st.session_state.insight_name == "Describe Table":
            st.subheader("Description Table")
            st.table(encoded_df.describe())
        # generating information table
        elif st.session_state.insight_name == "Info Table":
            st.subheader("Information Table")
            info_buffer = io.StringIO()
            encoded_df.info(buf=info_buffer)
            info_text = info_buffer.getvalue()
            st.text(info_text)
        # generating isnull table
        elif st.session_state.insight_name == "IsNull Table":
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
        name = title = st.text_input('Enter the name of your new column in which you want to add your selected column data and then press "Join Columns"', '')
        st.session_state.new_df = columns_join(encoded_df,name, join_columns)

        # join columns if statement
        if st.button("Join Columns"):
            st.subheader("Your new columns")
            st.table(st.session_state.new_df.head())

        st.subheader("Performing Feature Scaling")
        selected_features = st.multiselect('Select features for scaling and then press "Standardize" ', st.session_state.new_df.columns)

        if st.button("Standardize"):
            
            X = st.session_state.new_df[selected_features]
            y = st.session_state.new_df.drop(columns=selected_features).values

            # using sci-learn to split test and train data
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(X, y, test_size=0.33, random_state=369)

            # applying scaling to the model
            scaler = StandardScaler()
            st.session_state.X_train = scaler.fit_transform(st.session_state.X_train)
            st.session_state.X_test = scaler.transform(st.session_state.X_test)

            st.subheader("Standardized columns")
            st.table(st.session_state.X_test[:5])
        
        # generating graphs
        st.sidebar.header("Would you like to Visualize your new data now?")
        graph_name = st.sidebar.selectbox("Select Graphs to View",
                            ("","Heat Map", "Pair Plot", "Hist Plot"))
        if graph_name == 'Heat Map':
            st.subheader("Heat Map")
            fig, ax = plt.subplots()
            sns.heatmap(st.session_state.new_df.corr(), annot=True, ax=ax)
            st.pyplot(fig)
        elif graph_name == 'Pair Plot':
            st.subheader("Pair Plot")
            fig = sns.pairplot(st.session_state.new_df) 
            st.pyplot(fig)
        elif graph_name == 'Hist Plot':
            st.subheader("Hist Plot")
            fig,ax = plt.subplots()
            sns.histplot(st.session_state.new_df['price'], kde=True, stat="density", kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
            st.pyplot(fig)

        # generating regression models
        st.sidebar.header("Which Regression Model would you like to apply?")
        classifier_name = st.sidebar.selectbox("Select Regression Model",
                                        (" ","Linear Regression", "Random Forest", "XGBRF Regressor"))
        
        # regression models
        if classifier_name == 'Linear Regression':
            st.subheader("Linear Regression Result")
            regressor = LinearRegression()
            regressor.fit(st.session_state.X_train, st.session_state.y_train)
            y_pred = regressor.predict(st.session_state.X_train)
            plot_actual_vs_predicted(st.session_state.y_train, y_pred)
            find_metrics(st.session_state.y_train, y_pred)
        if classifier_name == 'Random Forest':
            st.subheader("Random Forest Result")
            regressor = RandomForestRegressor()
            regressor.fit(st.session_state.X_train, st.session_state.y_train)
            y_pred = regressor.predict(st.session_state.X_train)
            plot_actual_vs_predicted(st.session_state.y_train, y_pred)
            find_metrics(st.session_state.y_train, y_pred)            
        if classifier_name == 'XGBRF Regressor':
            st.subheader("XGBRF Regressor Result")
            regressor = XGBRFRegressor()
            regressor.fit(st.session_state.X_train, st.session_state.y_train)
            y_pred = regressor.predict(st.session_state.X_train)
            plot_actual_vs_predicted(st.session_state.y_train, y_pred)
            find_metrics(st.session_state.y_train, y_pred)                
            

if __name__ == '__main__':
    main()


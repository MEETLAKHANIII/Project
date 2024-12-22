import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest model
model = load('logistic_regression.joblib')

# Create a Streamlit app
st.title("Vehicle_Insured")

# Input fields for feature values on the main screen
st.header("Enter Customer Information")
tenure = st.number_input("response (in 0 and 1)", min_value=0, max_value=100, value=1)
internet_service = st.selectbox("Previously_Insured", ('DSL', 'Fiber optic', 'No'))
contract = st.selectbox("Vehicle_Age	", ('< 1 Year', '1-2 Year', '> 2 Years'))
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=200, value=50)
total_charges = st.number_input("Total Charges", min_value=0, max_value=10000, value=0)

# Map input values to numeric using the label mapping
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
}
internet_service = label_mapping[internet_service]
contract = label_mapping[contract]

# Make a prediction using the model
prediction = model.predict([[tenure, internet_service, contract, monthly_charges, total_charges]])

# Display the prediction result on the main screen
st.header("Prediction Result")
if prediction[0] == 0:
    st.success("This customer is likely to stay.")
else:
    st.error("This customer is likely to by Vehicle_Insured.")

# Add any additional Streamlit components or UI elements as needed.

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details below to predict whether the customer will churn.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=500.0)

# Convert Inputs into DataFrame
input_data = pd.DataFrame({
    "SeniorCitizen": [SeniorCitizen],
    "tenure": [tenure],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],
    "gender_Male": [1 if gender == "Male" else 0],
    "Partner_Yes": [1 if Partner == "Yes" else 0],
    "Dependents_Yes": [1 if Dependents == "Yes" else 0],
})

# Ensure correct scaling
scaled_data = scaler.transform(input_data)

if st.button("Predict Churn"):
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN!")
    else:
        st.success("✅ Customer is likely to STAY!")

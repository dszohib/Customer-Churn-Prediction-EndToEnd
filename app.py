import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")

st.write(
    "This application predicts whether a telecom customer is likely to churn "
    "based on account and service information."
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# User Input Section
# -----------------------------
st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2000.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

# -----------------------------
# Create Input Data
# -----------------------------
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "DeviceProtection": device_protection
}

input_df = pd.DataFrame([input_data])

# -----------------------------
# Preprocessing
# -----------------------------
# One-hot encoding
input_df = pd.get_dummies(input_df)

# Align columns with training data
input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scaling
scaled_data = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn")
    else:
        st.success(f"✅ Customer is likely to stay")

    st.write(f"Churn Probability: **{probability:.2f}**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Developed by **Zohib Khan**")
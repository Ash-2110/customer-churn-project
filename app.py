import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter Customer Details")

# Example important inputs (minimal demo version)
tenure = st.number_input("Tenure (Months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Create empty dataframe with training features
input_data = pd.DataFrame(columns=features)

# Fill only known features (others default to 0)
for col in features:
    input_data.loc[0, col] = 0

if "tenure" in features:
    input_data.loc[0, "tenure"] = tenure

if "MonthlyCharges" in features:
    input_data.loc[0, "MonthlyCharges"] = monthly_charges

if "TotalCharges" in features:
    input_data.loc[0, "TotalCharges"] = total_charges

if st.button("Predict"):

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn ❌")
    else:
        st.success("Customer will Stay ✅")
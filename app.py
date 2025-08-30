import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("loan_model.pkl", "rb"))

st.title("💰 Loan Default Prediction App")

st.write("Fill in the details below to check if a customer is likely to default.")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
loan_amount = st.number_input("Loan Amount ($)", min_value=100, value=1000)
past_defaults = st.number_input("Number of Past Defaults", min_value=0, value=0)

# Prediction
if st.button("Predict"):
    # Arrange input for the model
    features = np.array([[age, income, loan_amount, past_defaults]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ The applicant is **likely to default** on the loan.")
    else:
        st.success("✅ The applicant is **not likely to default** on the loan.")

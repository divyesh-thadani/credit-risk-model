import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
try:
    model = joblib.load('loan_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run `train_model.py` first to create it.")
    st.stop()

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# --- UI Elements ---
st.title('Loan Approval Prediction App 🏦')
st.write("This app predicts whether a loan application will be approved or denied based on the applicant's details.")

st.sidebar.header('Applicant Information')

# --- Function to get user input from the sidebar ---
def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    married = st.sidebar.selectbox('Married', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('0', '1', '2', '3+'))
    education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
    self_employed = st.sidebar.selectbox('Self Employed', ('Yes', 'No'))
    applicant_income = st.sidebar.number_input('Applicant Income', min_value=0, value=5000)
    coapplicant_income = st.sidebar.number_input('Coapplicant Income', min_value=0, value=1500)
    loan_amount = st.sidebar.number_input('Loan Amount (in thousands)', min_value=0, value=150)
    loan_amount_term = st.sidebar.number_input('Loan Amount Term (in months)', min_value=12, value=360, step=12)
    credit_history = st.sidebar.selectbox('Credit History', ('1.0 (Good)', '0.0 (Bad)'))
    property_area = st.sidebar.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))

    # Preprocess inputs to match the model's training format
    data = {
        'Gender': 1 if gender == 'Male' else 0,
        'Married': 1 if married == 'Yes' else 0,
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3}[dependents],
        'Education': 0 if education == 'Graduate' else 1,
        'Self_Employed': 1 if self_employed == 'Yes' else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': float(credit_history.split(' ')[0]),
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[property_area]
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs on the main page
st.subheader('Applicant Details')
st.write(input_df)

# Prediction button and logic
if st.button('Predict Loan Status'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display prediction result
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.success('Congratulations! Your loan is likely to be **Approved.**')
        st.write(f"Confidence: **{prediction_proba[0][1]*100:.2f}%**")
    else:
        st.error('We are sorry. Your loan is likely to be **Denied.**')
        st.write(f"Confidence: **{prediction_proba[0][0]*100:.2f}%**")

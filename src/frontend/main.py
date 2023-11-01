import streamlit as st
import requests
import json
import sys
sys.path.append("/Users/gscerberus/Desktop/Loan_Prediction_Analysis/")
from src.predictions import load_model
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import sys

print(sys.modules)

# Loading pickle files
encoder = load_model('serials/encoder.pkl')
scaler = load_model('serials/scaler.pkl')
model = load_model('serials/logistic_regression.pkl')

GENDER = encoder['gender']
MARRIED = encoder['married']
EDUCATION = encoder['education']
SELF_EMPLOYED = encoder['self_employed']
PROPERTY_AREA = encoder['property_area']
CREDIT_HISTORY = encoder['credit_history']

st.title("Will the applicant get his/her loan approved?")

gender = st.selectbox("Gender", GENDER)
married = st.selectbox("Married", MARRIED)
education = st.selectbox("Education", EDUCATION)
self_employed = st.selectbox("Self Employed", SELF_EMPLOYED)
property_area = st.selectbox("Property Area", PROPERTY_AREA)
credit_history = st.selectbox("Credit History", CREDIT_HISTORY)

total_income = st.slider("Total Income", 0, 500000, 10000)
loan_amount = st.slider("Loan Amount", 0,10000,1000)
loan_amount_term = st.slider("Loan Amount Term",0, 360, 40)

# parsing inputs to json file format so they can get passed to rest api
inputs = {
    "Gender":gender,
    "Married":married,
    "Education":education,
    "Self_Employed":self_employed,
    "LoanAmount":loan_amount,
    "Loan_Amount_Term":loan_amount_term,
    "Credit_History":credit_history,
    "Property_Area":property_area,
    "TotalIncome": total_income
}

# We should not forget to run them simultaneously
predict_button = st.button("Check Status")
if predict_button:

    res = requests.post(url = "http://0.0.0.0:8000/predict", data = json.dumps(inputs))
    # Check if the request was successful (status code 200)
    if res.status_code == 200:
        prediction = res.json()[0]

        if prediction == 0:
            st.subheader("The loan can't be approved!")
        if prediction == 1:
            st.subheader("The loan can be approved!")

        #get_res = requests.get(url = "http://0.0.0.0:8080/get_result")
        #if get_res.status_code == 200:
         #   fetched_result = get_res.json()["result"]








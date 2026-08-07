import requests
import streamlit as st
import os


st.title("Churn Predictor")
st.write("Finds the approx precentage of customer leaving company's services")

monthly_charges = st.slider("Monthly Charge ($)", min_value=1, max_value=200, value=30)
contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
tenure = st.slider("Tenure in Months", min_value=1, max_value=120, value=1)
payment_method = st.selectbox("Select Payment Method", ["Credit Card", "Bank Withdrawal"])
has_internet = st.selectbox("Internet Service", ["Yes", "No"])

internet_type = None
if has_internet == "Yes":
    internet_type = st.selectbox("Internet Type", ["DSL", "Cable", "Fiber Optic"])
else:
    internet_type = "No"

if st.button("Predict Churn %"):
    payload = {
        "monthly_charges": monthly_charges,
        "tenure": tenure,
        "contract": contract,
        "payment_method": payment_method,
        "has_internet": has_internet,
        "internet_type": internet_type
    }

    try:
        BACKEND_URL = os.getenv("BACKEND_URL", "https://churn-api-backend.onrender.com/predict")
        response = requests.post(BACKEND_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.metric(label="Churn Probability", value=f"{result['churn_percentage']}%")
        else:
            st.error("Server error during prediction.")
    except Exception as e:
        st.error("Could not connect to FastAPI server. Make sure it is running!")





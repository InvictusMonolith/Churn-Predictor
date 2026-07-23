import pandas as pd
import joblib
import streamlit as st

numeric_defaults = joblib.load("numeric_defaults.joblib")
model = joblib.load("churn_model.joblib")
churn = joblib.load("churn_scaler.joblib")

MODEL_COLUMNS = ['Age',
 'Avg Monthly GB Download',
 'Avg Monthly Long Distance Charges',
 'CLTV',
 'Dependents',
 'Device Protection Plan',
 'Internet Service',
 'Latitude',
 'Longitude',
 'Married',
 'Monthly Charge',
 'Multiple Lines',
 'Number of Dependents',
 'Number of Referrals',
 'Online Backup',
 'Online Security',
 'Paperless Billing',
 'Partner',
 'Phone Service',
 'Population',
 'Premium Tech Support',
 'Referred a Friend',
 'Satisfaction Score',
 'Senior Citizen',
 'Streaming Movies',
 'Streaming Music',
 'Streaming TV',
 'Tenure in Months',
 'Total Charges',
 'Total Extra Data Charges',
 'Total Long Distance Charges',
 'Total Refunds',
 'Total Revenue',
 'Under 30',
 'Unlimited Data',
 'Contract_Month-to-Month',
 'Contract_One Year',
 'Contract_Two Year',
 'Gender_Female',
 'Gender_Male',
 'Internet Type_Cable',
 'Internet Type_DSL',
 'Internet Type_Fiber Optic',
 'Offer_Offer A',
 'Offer_Offer B',
 'Offer_Offer C',
 'Offer_Offer D',
 'Offer_Offer E',
 'Payment Method_Bank Withdrawal',
 'Payment Method_Credit Card',
 'Payment Method_Mailed Check'

]

st.title("Churn Predictor")
st.write("Finds the approx percentage of customer leaving company's services")

monthly_charges = st.slider(
    "Monthly Charge ($)",
    min_value=1,
    max_value=200,
    value=30

)

contract = st.selectbox(
    "Contract Type" ,
    ["Month-to-Month" , "One Year" , "Two Year"]
)
contract_one_year = 1 if contract== "One Year" else 0 
contract_Two_year = 1 if contract== "Two Year" else 0 

tenure = st.slider(
    "Tenure in Months" , 
    min_value=1,
    max_value=120,
    value=1
)

Payment_method = st.selectbox(
    "Select Payment Method",
    ["Credit Card" , "Bank Withdrawal"]

)
has_internet = st.selectbox(
    "Internet Service",
    ["Yes", "No"]
)

if has_internet == "Yes":
    internet_type = st.selectbox(
        "Internet Type",
        ["DSL", "Cable", "Fiber Optic"]
    )

if st.button("Predict Churn %"):

    input_dict = {col: 0 for col in MODEL_COLUMNS}
    input_dict.update(numeric_defaults)

    # Internet Service
    input_dict["Internet Service"] = 1 if has_internet == "Yes" else 0

    if has_internet == "Yes":
        input_dict[f"Internet Type_{internet_type}"] = 1

    # Other inputs
    input_dict["Monthly Charge"] = monthly_charges
    input_dict["Tenure in Months"] = tenure
    input_dict["Contract_Month-to-Month"] = 1 if contract == "Month-to-Month" else 0

    input_dict["Contract_One Year"] = 1 if contract == "One Year" else 0
    input_dict["Contract_Two Year"] = 1 if contract == "Two Year" else 0

    input_dict["Payment Method_Credit Card"] = (
        1 if Payment_method == "Credit Card" else 0
    )

    input_dict["Payment Method_Bank Withdrawal"] = (
        1 if Payment_method == "Bank Withdrawal" else 0
    )

    input_data = pd.DataFrame([input_dict], columns=MODEL_COLUMNS)

    input_scaled = churn.transform(input_data)

    proba = model.predict_proba(input_scaled)

    prediction = proba[0][1] * 100

    st.metric(
        label="Churn Probability",
        value=f"{prediction:.2f}%"
    )



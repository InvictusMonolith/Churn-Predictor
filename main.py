from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Predictor API! Go to /docs to test it."} 

numeric_defaults = joblib.load("models/numeric_defaults.joblib")
model = joblib.load("models/churn_model.joblib")
scaler = joblib.load("models/churn_scaler.joblib")

MODEL_COLUMNS = [
    'Age', 'Avg Monthly GB Download', 'Avg Monthly Long Distance Charges', 'CLTV',
    'Dependents', 'Device Protection Plan', 'Internet Service', 'Latitude',
    'Longitude', 'Married', 'Monthly Charge', 'Multiple Lines', 'Number of Dependents',
    'Number of Referrals', 'Online Backup', 'Online Security', 'Paperless Billing',
    'Partner', 'Phone Service', 'Population', 'Premium Tech Support',
    'Referred a Friend', 'Satisfaction Score', 'Senior Citizen', 'Streaming Movies',
    'Streaming Music', 'Streaming TV', 'Tenure in Months', 'Total Charges',
    'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Refunds',
    'Total Revenue', 'Under 30', 'Unlimited Data', 'Contract_Month-to-Month',
    'Contract_One Year', 'Contract_Two Year', 'Gender_Female', 'Gender_Male',
    'Internet Type_Cable', 'Internet Type_DSL', 'Internet Type_Fiber Optic',
    'Offer_Offer A', 'Offer_Offer B', 'Offer_Offer C', 'Offer_Offer D',
    'Offer_Offer E', 'Payment Method_Bank Withdrawal', 'Payment Method_Credit Card',
    'Payment Method_Mailed Check'
]

class ChurnInput(BaseModel):
    monthly_charges: float
    tenure: int
    contract: Literal["Month-to-Month", "One Year", "Two Year"]
    payment_method: Literal["Credit Card", "Bank Withdrawal"]
    has_internet: Literal["Yes", "No"]
    internet_type: Literal["DSL", "Cable", "Fiber Optic", "No"]

@app.post("/predict")
def predict(data: ChurnInput):
    input_dict = {col: 0 for col in MODEL_COLUMNS}
    input_dict.update(numeric_defaults)

    # Inject UI values
    input_dict["Monthly Charge"] = data.monthly_charges
    input_dict["Tenure in Months"] = data.tenure
    input_dict["Internet Service"] = 1 if data.has_internet == "Yes" else 0

    if data.has_internet == "Yes" and data.internet_type:
        key = f"Internet Type_{data.internet_type}"
        if key in input_dict:
            input_dict[key] = 1

    input_dict["Contract_Month-to-Month"] = 1 if data.contract == "Month-to-Month" else 0
    input_dict["Contract_One Year"] = 1 if data.contract == "One Year" else 0
    input_dict["Contract_Two Year"] = 1 if data.contract == "Two Year" else 0

    input_dict["Payment Method_Credit Card"] = 1 if data.payment_method == "Credit Card" else 0
    input_dict["Payment Method_Bank Withdrawal"] = 1 if data.payment_method == "Bank Withdrawal" else 0

    # Calculate Total Charges dynamically so model doesn't get corrupted
    input_dict["Total Charges"] = data.monthly_charges * data.tenure

    # Scale and predict
    input_data = pd.DataFrame([input_dict], columns=MODEL_COLUMNS)
    input_scaled = scaler.transform(input_data)
    proba = model.predict_proba(input_scaled)[0][1] * 100

    return {"churn_percentage": round(proba, 2)}
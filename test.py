from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_valid_prediction():
    response = client.post("/predict", json={
        "monthly_charges": 138,
        "tenure": 89,
        "contract": "Two Year",
        "payment_method": "Credit Card",
        "has_internet": "Yes",
        "internet_type": "DSL"
    })
    assert response.status_code == 200
    body = response.json()
    assert "churn_percentage" in body
    assert 0 <= body["churn_percentage"] <= 100


def test_invalid_contract_rejected():
    response = client.post("/predict", json={
        "monthly_charges": 100,
        "tenure": 5,
        "contract": "Three Year",
        "payment_method": "Credit Card",
        "has_internet": "Yes",
        "internet_type": "DSL"
    })
    assert response.status_code == 422


def test_invalid_payment_method_rejected():
    response = client.post("/predict", json={
        "monthly_charges": 100,
        "tenure": 5,
        "contract": "Month-to-Month",
        "payment_method": "Cash",
        "has_internet": "Yes",
        "internet_type": "DSL"
    })
    assert response.status_code == 422


def test_missing_field_rejected():
    response = client.post("/predict", json={
        "monthly_charges": 100,
        "contract": "Month-to-Month",
        "payment_method": "Credit Card",
        "has_internet": "Yes",
        "internet_type": "DSL"
        # "tenure" intentionally missing
    })
    assert response.status_code == 422
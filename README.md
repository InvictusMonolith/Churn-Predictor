# 📊 Telco Customer Churn Predictor

A Logistic Regression model that predicts the probability of a telecom customer churning, served through an interactive Streamlit web app.

Built as part of a BSc Data Science project to practice end-to-end ML: data cleaning → feature engineering → model training → serialization → deployment via UI.

---

## 🎯 What it does

Given a customer's contract type, tenure, monthly charge, internet service, and payment method, the app returns the **probability (0–100%) that the customer will churn** (leave the service).

---

## 📈 Model Performance

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|-------|----------|-------------------|----------------|------------|
| **Logistic Regression** (deployed) | 0.95 | 0.86 | 0.96 | 0.91 |
| Random Forest (comparison) | 0.97 | 0.99 | 0.91 | 0.94 |

Logistic Regression was chosen for deployment because its high recall on the minority (churn) class is the most useful business signal — better to flag a customer who's *at risk* and act on it than to miss them.

Class imbalance was handled with `class_weight="balanced"`.

---

## 🗂️ Dataset

**Telco_customer_churn_dataset.csv** — IBM Telco Customer Churn dataset (~7,000 customers, 33 features after preprocessing).

Dropped columns (data leakage / identifiers / low signal):
- `Churn Category`, `Churn Reason`, `Churn Score` (leak the target)
- `Customer ID`, `City`, `State`, `Country`, `Lat Long`, `Zip Code` (identifiers / high cardinality)
- `Customer Status` (post-outcome label)
- `Quarter` (only one unique value)

Categorical features one-hot encoded: `Contract`, `Gender`, `Internet Type`, `Offer`, `Payment Method`.

Class distribution: **73.5% non-churn, 26.5% churn** (mild imbalance).

---

## 🛠️ Tech Stack

- **Python 3.x**
- **scikit-learn** — Logistic Regression, StandardScaler, train/test split
- **pandas / numpy** — data wrangling
- **streamlit** — interactive UI
- **joblib** — model + scaler + numeric defaults persistence

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/churn-predictor.git
cd churn-predictor

2. Install dependencies

pip install -r requirements.txt

3. Train the model (optional — pre-trained models included)

Open notebooks/Untitled.ipynb in Jupyter and run all cells. This produces:
- models/churn_model.joblib
- models/churn_scaler.joblib
- models/numeric_defaults.joblib

4. Launch the app

streamlit run app/Churn_pro.py

The app will open at http://localhost:8501.

---
🧠 How the App Works

1. User sets the customer's Monthly Charge, Tenure, Contract Type, Payment Method, and Internet Service/Type via sliders and dropdowns.
2. The app builds a one-hot-encoded input row matching the model's training columns (MODEL_COLUMNS).
3. Any field the user doesn't set is filled with the training-set mean for that numeric feature, and 0 for unused one-hot columns.
4. The input is scaled with the same StandardScaler used in training.
5. The Logistic Regression model outputs a churn probability, displayed as a percentage.

---
🐛 Notable Bug & Fix

Symptom: The app returned ~100% churn probability for almost every input, even clearly low-risk profiles (long-tenure Two-Year contract customers, low monthly charges).

Root cause: The numeric_defaults dictionary was being constructed from the raw training DataFrame, which had True/False (boolean) values for the one-hot-encoded columns. When the app loaded these defaults and merged them into the input row, boolean values were being passed where the model expected numerics, causing the scaler to misbehave and the predictions to saturate.

Fix: Compute numeric_defaults from X_train excluding one-hot columns — only mean-fill the actual continuous numeric features (Age, Latitude, Longitude, CLTV, Total Charges, etc.). The one-hot columns are explicitly set to 0 or 1 based on user input.

Result: Low-risk profile → 7.23%, High-risk profile → 77.02%. Strong, plausible separation across the spectrum.

---
✅ Verification

Two sanity-check profiles have been run through the app to confirm the model is wired correctly:

┌────────────────────────────────────────────────────────────────────────────┬────────────────┬────────┬─────────┐
│                                  Profile                                   │    Expected    │  Got   │ Verdict │
├────────────────────────────────────────────────────────────────────────────┼────────────────┼────────┼─────────┤
│ Two-Year contract, 25-month tenure, low monthly charge, bank withdrawal    │ Low (~5–10%)   │ 7.23%  │ ✅      │
├────────────────────────────────────────────────────────────────────────────┼────────────────┼────────┼─────────┤
│ Month-to-Month, 1-month tenure, $166 monthly, bank withdrawal, no internet │ High (~70–85%) │ 77.02% │ ✅      │
└────────────────────────────────────────────────────────────────────────────┴────────────────┴────────┴─────────┘

A single-variable isolation test (change only contract type in the high-risk profile to "Two Year" and confirm the probability drops sharply) is the next step to fully verify.

---
📁 Project Structure

churn-predictor/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files to ignore in version control
├── app/
│   └── Churn_pro.py             # Streamlit app
├── notebooks/
│   └── Untitled.ipynb           # Model training notebook
├── models/                      # Pre-trained artifacts
│   ├── churn_model.joblib       # Logistic Regression model
│   ├── churn_scaler.joblib      # StandardScaler
│   └── numeric_defaults.joblib  # Numeric feature means (for default fill)
└── data/
    └── Telco_customer_churn_dataset.csv

---
📝 License

Dataset: IBM Telco Customer Churn (https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) (Kaggle, public).

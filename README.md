# 📉 Telco Customer Churn Predictor

A machine learning project that predicts the probability of a telecom customer churning — built end-to-end from raw customer data to a containerized, publicly deployed prediction API with a live interface.

**[Live API](#)** *(https://churn-api-backend.onrender.com)*


**[Live Demo](#)** *(https://predictor-churn.streamlit.app/)*

![App Screenshot](#) *(<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2996d3e7-59ff-4878-a104-8865c74dffbb" />
)*

Note: the API is hosted on Render's free tier, which spins down after inactivity — the first prediction after a period of no traffic may take 20-50 seconds while the service wakes up. Subsequent requests are fast.

---

## What This Project Does

Enter a customer's contract type, tenure, monthly charges, internet service, and payment method — get back a churn probability, served through a validated, containerized REST API and displayed in an interactive Streamlit interface.

---

## Dataset

- **Source:** IBM Telco Customer Churn dataset (via Hugging Face).
- **Size:** 4,225 customer records, 52 original columns.
- **Target:** `Churn` (binary — 0 = stayed, 1 = churned). Class distribution: ~73.5% stayed, ~26.5% churned.

### Data Leakage Found and Removed
Three sources of leakage had to be identified and dropped before training — not because they were low-quality, but because they were derived from or duplicated the target itself:

| Column | Why it's leakage |
|---|---|
| `Churn Score` | A pre-calculated churn risk score from IBM's own predictive tool — training on it would mean learning to copy another model's output, not learning from customer data. |
| `Churn Category` / `Churn Reason` | Only populated for customers who already churned — 73.5% missing exactly matches the non-churned population. |
| `Customer Status` | Near-duplicate of the target itself (`Stayed`/`Churned`/`Joined`). |

Also dropped: `Country` and `State` (zero variance), `City` (1,085 unique values — too high-cardinality relative to dataset size; `Latitude`/`Longitude` retained instead).

---

## Modeling

Two models were trained and compared, using `class_weight="balanced"` to address class imbalance:

| Model | Accuracy | Churn Recall | Churn Precision |
|---|---|---|---|
| **Logistic Regression (final)** | 0.95 | **0.96** | 0.86 |
| Random Forest | 0.97 | 0.91 | 0.99 |

### Why Logistic Regression Was Chosen Over the Higher-Accuracy Model
Random Forest scores higher on raw accuracy and precision, but Logistic Regression catches more actual churners (96% recall vs. 91%) — missing 8 churned customers instead of 21. For a churn model, a missed churner (false negative) is typically more costly than a false alarm (false positive): a false alarm costs an unnecessary retention email, while a missed churner is lost revenue with no chance to intervene. Model selection here was driven by the business cost of each error type, not just the headline metric.

---

## Architecture

```
Streamlit UI  →  HTTP POST  →  FastAPI /predict (Dockerized, deployed on Render)  →  Model + Scaler  →  JSON response
```

- **Input validation:** fields like `contract`, `payment_method`, `has_internet`, and `internet_type` are constrained with Pydantic `Literal` types, so invalid values (e.g. `"contract": "Three Year"`) are rejected with a `422` error before ever reaching the model.
- **Automated tests:** a `pytest` suite covers valid predictions, invalid field rejection, and missing-field rejection. Verified by intentionally breaking a `Literal` constraint and confirming the corresponding test fails — not just that tests exist, but that they catch real regressions.
- **Containerized:** packaged with a `Dockerfile` so the API runs identically regardless of host environment, and deployed live on Render.

**Scope note:** the UI collects the most predictive fields (tenure, contract, monthly charges, internet service/type, payment method). The remaining features the model was trained on default to their training-set means — a deliberate scope decision to keep the interface usable, documented here rather than hidden.

---

## Tech Stack

- **Python** — pandas, scikit-learn, joblib
- **FastAPI** + **Pydantic** — validated prediction API
- **pytest** — automated endpoint testing
- **Docker** — containerized deployment
- **Render** — live hosting
- **Streamlit** — client interface
- **Jupyter / Anaconda** — data cleaning and model development

---

## How to Run

### Option 1 — Docker (recommended, matches production)
```bash
git clone https://github.com/InvictusMonolith/Churn-Predictor.git
cd Churn-Predictor
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
API docs available at `http://127.0.0.1:8000/docs`

### Option 2 — Local Python environment
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Running the Streamlit client
```bash
streamlit run Churn_pro.py
```

### Running tests
```bash
pytest -v test.py
```

---

## Project Structure
```
├── ChurnPredictor.ipynb        # Data cleaning, model training, evaluation
├── main.py                      # FastAPI prediction endpoint with input validation
├── Churn_pro.py                 # Streamlit client interface
├── test.py                      # pytest suite for the API
├── Dockerfile                   # Container definition
├── docker-compose.yml
├── requirements.txt
├── models/
├── churn_model.joblib            # Trained Logistic Regression model
├── churn_scaler.joblib           # Fitted StandardScaler
├── numeric_defaults.joblib       # Training-mean fallback values for unused inputs
└── README.md
```

---

## What I Learned

- **Class imbalance changes the evaluation strategy, not just the model.** Accuracy alone would have hidden the real story here — precision/recall/F1 per class, and the false-negative vs. false-positive cost tradeoff, mattered more than picking the model with the highest single number.
- **Leakage isn't always an obvious duplicate column.** `Churn Score` looked like a legitimate feature at first glance; only checking the dataset's documentation revealed it was another model's output sitting in the training data.
- **A silent zero default is dangerous for linear models in a way it isn't for tree models.** Zero-filling unused numeric fields caused predictions to saturate near 100% once scaled — fixed by defaulting unused fields to training-set means instead of zero.
- **A passing test isn't proof of anything by itself.** The endpoint validation tests were only trustworthy after intentionally breaking the validation logic and confirming the corresponding test failed — proving the test catches real regressions, not just that it runs without error.
- **Containerizing exposes assumptions a local environment hides.** Getting the Dockerfile to correctly resolve model file paths required being explicit about exactly what gets copied into the image and where — something that "just works" locally can silently fail once the filesystem context changes.

---

## Roadmap

- [x] Data cleaning and leakage removal
- [x] Model comparison with business-cost-aware model selection
- [x] Streamlit prediction interface
- [x] FastAPI validated prediction endpoint
- [x] Automated test suite (verified against real regressions)
- [x] Dockerized deployment
- [x] Live deployment on Render
- [ ] CI pipeline to auto-run tests on push

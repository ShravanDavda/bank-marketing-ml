# Bank Marketing Predictor — Web Interface (Horizontal Layout)

This document provides complete instructions for running, testing, and using the **Bank Marketing Predictor** web application built with Streamlit.

The application features a modern, desktop-first **horizontal two-column workspace**:
- **Left Column (65% width)**: Compact customer profile and campaign input form.
- **Right Column (35% width)**: Persistent prediction panel with an initial empty state, real-time prediction results card (positive/neutral indicator and horizontal probability meter), and a secondary model specifications card.

---

## 1. Prerequisites and Environment

The application requires Python 3.9+ and the following packages:

- `streamlit >= 1.30.0`
- `pandas >= 2.0.0`
- `scikit-learn >= 1.3.0`
- `joblib >= 1.3.0`
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`

### Using the Project-Local Virtual Environment

If using the existing `.venv` environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

If setting up a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install streamlit joblib
```

---

## 2. Model Artifact and Generation

The web application reuses the project's exact final tuned Random Forest pipeline:

- **Artifact Path**: `models/random_forest_pipeline.joblib`
- **Pipeline Components**:
  1. `ColumnTransformer` preprocessor (StandardScaler for numerical features, OneHotEncoder for categoricals).
  2. `RandomForestClassifier` (200 estimators, max depth 15, min samples leaf 3, balanced class weight, random state 42).
- **Pre-contact Constraint**: The `duration` feature is strictly excluded from model inputs.

### Exporting / Re-generating the Model

If the artifact is missing or needs to be regenerated:

```bash
python export_model.py
```

This script:
1. Loads `data/bank-full.csv`.
2. Applies feature engineering (`previously_contacted = (pdays != -1)`).
3. Fits the complete preprocessing and tuned Random Forest pipeline on training data.
4. Validates test metrics (F1 ~ 0.46–0.48, ROC-AUC ~ 0.802).
5. Serializes the fitted pipeline to `models/random_forest_pipeline.joblib`.

---

## 3. How to Launch the Application

From the project root (`bank-marketing-ml/`), run:

```bash
streamlit run app.py
```

Or directly using the virtual environment executable:

```powershell
.\.venv\Scripts\streamlit run app.py
```

Once launched, open your web browser at:
`http://localhost:8501`

---

## 4. Input Field Descriptions

The single-page form is structured into three clean sections:

### Section 1: Personal Profile
- **Customer Age**: Age in years (18 to 100).
- **Occupation (Job)**: Job category (`management`, `technician`, `admin.`, `blue-collar`, etc.).
- **Marital Status**: Marital status (`married`, `single`, `divorced`).
- **Education Level**: Highest education completed (`primary`, `secondary`, `tertiary`, `unknown`).

### Section 2: Financial Details
- **Average Yearly Balance**: Account balance in Euros (supports negative balances).
- **Credit in Default**: Whether customer has credit currently in default (`no` / `yes`).
- **Housing Loan**: Whether customer has an active mortgage/housing loan (`no` / `yes`).
- **Personal Loan**: Whether customer has an active personal loan (`no` / `yes`).

### Section 3: Campaign Information
- **Contact Communication Type**: Cellular, telephone, or unknown.
- **Contact Month & Day**: Planned month and day for the contact.
- **Contacts During Current Campaign**: Number of contacts during this campaign (minimum 1).
- **Prior Campaign Contact**:
  - Select **No** for customers with no recorded prior campaign contacts (`pdays` automatically set to `-1`, `previous` to `0`, `poutcome` to `unknown`).
  - Select **Yes** to input the number of days since prior contact, previous contacts count, and previous outcome (`failure`, `other`, `success`, `unknown`).

---

## 5. Interpreting Predictions

Clicking **Predict Subscription** evaluates the single-row customer record and renders a dedicated result card:

1. **Outcome**:
   - **Likely to subscribe** (Positive class: `'yes'`): The customer exhibits characteristics aligned with term deposit subscribers. Highlighted in soft green (`#F0FDF4`).
   - **Unlikely to subscribe** (Negative class: `'no'`): The customer is less likely to subscribe based on historical data. Rendered in a clean, neutral slate tone (`#F8FAFC`).
2. **Estimated Subscription Probability & Visual Meter**:
   - The probability that the customer belongs to the positive class (`'yes'`), extracted from `predict_proba()`.
   - A clean horizontal probability meter displays the exact percentage relative to the 50% decision baseline, providing immediate visual feedback without noisy charts.
3. **Model**:
   - Displays the verified model name: `Random Forest (Tuned Pipeline)`.
4. **Disclaimer**:
   - *"Predictions are estimates based on historical data and do not guarantee actual customer behavior."*

---

## 6. Known Limitations

- **Imbalanced Dataset**: The original dataset contains ~11.7% actual subscribers. The model is tuned with balanced class weighting to maximize detection (recall) at the expense of lower precision.
- **Pre-contact Constraint**: Because duration of the conversation is not known beforehand, `duration` is deliberately excluded. This results in realistic pre-campaign predictions, avoiding data leakage.
- **Estimates Only**: Predictions reflect statistical correlations and do not guarantee customer conversion.

---

## 7. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Model Artifact Missing` | `models/random_forest_pipeline.joblib` does not exist | Run `python export_model.py` to train and save the pipeline. |
| `ModuleNotFoundError: streamlit` | Streamlit is not installed in the active Python environment | Run `.\.venv\Scripts\pip install streamlit`. |
| Port 8501 is already in use | Another Streamlit process is running | Run `streamlit run app.py --server.port 8502`. |

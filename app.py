"""
Bank Marketing Predictor — Streamlit Web Application.
Horizontal Two-Column Layout (Desktop-First Workspace).

Layout:
- Left Column (65% width): Customer profile and campaign input form.
- Right Column (35% width): Real-time prediction results panel, probability meter,
  and model metadata.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    DEFAULT_MODEL_PATH,
    VALID_CATEGORIES,
    build_inference_dataframe,
    load_trained_model,
    predict_subscription,
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup — Wide Horizontal Workspace
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Refined Modern Two-Column Layout Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global Page Background & Typography */
    .stApp {
        background-color: #F7F9FC !important;
        color: #172B4D !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Wide Desktop Workspace Alignment (approx 1280–1350px) */
    .block-container {
        max-width: 1320px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Universal Card Containers for Left Form Sections */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E1E7F0 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 1px 3px rgba(9, 30, 66, 0.05) !important;
    }

    /* Compact Header Card */
    .header-container {
        background-color: #FFFFFF;
        border: 1px solid #E1E7F0;
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(9, 30, 66, 0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .header-icon {
        font-size: 30px;
        line-height: 1;
    }

    .app-title {
        color: #172B4D;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.2px;
    }

    .app-subtitle {
        color: #5E6C84;
        font-size: 13.5px;
        margin: 2px 0 0 0;
        line-height: 1.3;
    }

    .header-badge-group {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: #EBF3FF;
        color: #2457B8;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        letter-spacing: 0.3px;
        border: 1px solid #D0E2FF;
    }

    .model-indicator {
        font-size: 12px;
        color: #5E6C84;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 4px 10px;
        border-radius: 6px;
    }

    .model-dot {
        width: 7px;
        height: 7px;
        background-color: #36B37E;
        border-radius: 50%;
        display: inline-block;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #EDF1F7;
    }

    .section-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        background-color: #F0F4FA;
        color: #2457B8;
        font-size: 11px;
        font-weight: 700;
        border-radius: 50%;
    }

    .section-heading-text {
        font-size: 15px;
        font-weight: 700;
        color: #172B4D;
        margin: 0;
    }

    /* Labels Above Inputs */
    div[data-testid="stWidgetLabel"] label p {
        color: #172B4D !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important;
    }

    /* Input Controls */
    .stTextInput input, .stNumberInput input, div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-color: #DCE3ED !important;
        border-radius: 8px !important;
        color: #172B4D !important;
        font-size: 13.5px !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #2457B8 !important;
        box-shadow: 0 0 0 1px #2457B8 !important;
    }

    /* Action Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #2457B8 !important;
        border: 1px solid #2457B8 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
        border-radius: 8px !important;
        padding: 9px 20px !important;
        box-shadow: 0 1px 2px rgba(36, 87, 184, 0.2) !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1B4596 !important;
        border-color: #1B4596 !important;
    }

    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3ED !important;
        color: #4A5568 !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
        border-radius: 8px !important;
        padding: 9px 20px !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #172B4D !important;
    }

    /* Right Column Prediction Cards */
    .results-panel-box {
        background-color: #FFFFFF;
        border: 1px solid #E1E7F0;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 1px 3px rgba(9, 30, 66, 0.05);
        margin-bottom: 16px;
    }

    .empty-state-card {
        text-align: center;
        padding: 30px 18px;
    }

    .empty-icon-wrap {
        width: 52px;
        height: 52px;
        background-color: #F0F4FA;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 14px;
    }

    .empty-heading {
        font-size: 17px;
        font-weight: 700;
        color: #172B4D;
        margin: 0 0 8px 0;
    }

    .empty-text {
        font-size: 13.5px;
        color: #5E6C84;
        margin: 0 auto 16px auto;
        line-height: 1.45;
        max-width: 290px;
    }

    /* Active Prediction Cards */
    .result-box-positive {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 1px 3px rgba(16, 185, 129, 0.08);
        margin-bottom: 16px;
    }

    .result-box-neutral {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 1px 3px rgba(9, 30, 66, 0.05);
        margin-bottom: 16px;
    }

    .result-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .result-badge-positive {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #DCFCE7;
        color: #15803D;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 5px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .result-badge-neutral {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #F1F5F9;
        color: #475569;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 5px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .result-outcome-text {
        font-size: 20px;
        font-weight: 700;
        margin: 0;
    }

    .positive-text {
        color: #15803D;
    }

    .neutral-text {
        color: #334155;
    }

    .result-detail-text {
        font-size: 13.5px;
        color: #475569;
        margin: 4px 0 14px 0;
    }

    /* Horizontal Probability Meter */
    .meter-container {
        margin: 14px 0 10px 0;
    }

    .meter-track {
        height: 10px;
        background-color: #E2E8F0;
        border-radius: 6px;
        overflow: hidden;
    }

    .meter-fill-positive {
        height: 100%;
        background-color: #16A34A;
        border-radius: 6px;
        transition: width 0.3s ease;
    }

    .meter-fill-neutral {
        height: 100%;
        background-color: #475569;
        border-radius: 6px;
        transition: width 0.3s ease;
    }

    .meter-labels {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #64748B;
        margin-top: 5px;
        font-weight: 500;
    }

    .result-footer-disclaimer {
        font-size: 11.5px;
        color: #64748B;
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px dashed #CBD5E1;
        line-height: 1.4;
    }

    /* Secondary Model Info Card */
    .model-info-card {
        background-color: #FFFFFF;
        border: 1px solid #E1E7F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(9, 30, 66, 0.04);
    }

    .info-card-title {
        font-size: 13px;
        font-weight: 700;
        color: #172B4D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #EDF1F7;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        font-size: 12.5px;
        padding: 4px 0;
    }

    .info-label {
        color: #64748B;
    }

    .info-val {
        color: #172B4D;
        font-weight: 600;
    }

    /* Callout */
    .custom-info-callout {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12.5px;
        color: #475569;
        margin-top: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning pipeline...")
def get_cached_model():
    """
    Load the trained pipeline once into Streamlit memory cache.
    """
    return load_trained_model(DEFAULT_MODEL_PATH)


# -----------------------------------------------------------------------------
# Form Session State Management
# -----------------------------------------------------------------------------
DEFAULT_FORM_VALUES = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "secondary",
    "balance": 1000.0,
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day": 15,
    "campaign": 1,
    "has_prior_contact": "No",
    "pdays_val": 90,
    "previous_val": 1,
    "poutcome_val": "unknown",
    "prediction_result": None,
}


def init_state():
    for key, val in DEFAULT_FORM_VALUES.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


def do_reset():
    for key, val in DEFAULT_FORM_VALUES.items():
        st.session_state[key] = val


# -----------------------------------------------------------------------------
# Compact Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-container">
        <div class="header-left">
            <span class="header-icon">🏦</span>
            <div>
                <h1 class="app-title">Bank Marketing Predictor</h1>
                <p class="app-subtitle">Estimate the likelihood of a customer subscribing to a term deposit.</p>
            </div>
        </div>
        <div class="header-badge-group">
            <span class="model-indicator">
                <span class="model-dot"></span> Random Forest · Predictive Model
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model Verification on Startup
# -----------------------------------------------------------------------------
model_loaded = False
model = None

try:
    model = get_cached_model()
    model_loaded = True
except FileNotFoundError:
    st.error(
        f"**Model Artifact Missing**: The trained model file was not found at `{DEFAULT_MODEL_PATH}`.\n\n"
        "To generate the model artifact, run the export script from your terminal:\n\n"
        "```bash\npython export_model.py\n```"
    )
except Exception as e:
    st.error(f"**Error Loading Model**: {e}")


# -----------------------------------------------------------------------------
# Main Horizontal Two-Column Layout (65% Form Left, 35% Results Right)
# -----------------------------------------------------------------------------
col_form, col_panel = st.columns([65, 35], gap="large")

with col_form:
    # -------------------------------------------------------------------------
    # Customer Details Form (Left Column)
    # -------------------------------------------------------------------------
    with st.form("bank_marketing_form", border=False):

        # Section 1: Personal Profile
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-header">
                    <span class="section-num">1</span>
                    <span class="section-heading-text">Personal Profile</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_p1, col_p2 = st.columns(2)

            with col_p1:
                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=int(st.session_state["age"]),
                    step=1,
                    help="Customer age in years (18 to 100).",
                )
                job = st.selectbox(
                    "Job / Occupation",
                    options=VALID_CATEGORIES["job"],
                    index=VALID_CATEGORIES["job"].index(st.session_state["job"]),
                    format_func=lambda x: "Admin" if x == "admin." else x.replace("-", " ").title(),
                    help="Customer occupation category.",
                )

            with col_p2:
                marital = st.selectbox(
                    "Marital Status",
                    options=VALID_CATEGORIES["marital"],
                    index=VALID_CATEGORIES["marital"].index(st.session_state["marital"]),
                    format_func=lambda x: x.title(),
                    help="Customer marital status.",
                )
                education = st.selectbox(
                    "Education Level",
                    options=VALID_CATEGORIES["education"],
                    index=VALID_CATEGORIES["education"].index(st.session_state["education"]),
                    format_func=lambda x: x.title(),
                    help="Highest level of completed education.",
                )

        # Section 2: Financial Details
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-header">
                    <span class="section-num">2</span>
                    <span class="section-heading-text">Financial Details</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_f1, col_f2 = st.columns(2)

            with col_f1:
                balance = st.number_input(
                    "Average Yearly Balance (€)",
                    min_value=-10000.0,
                    max_value=200000.0,
                    value=float(st.session_state["balance"]),
                    step=100.0,
                    format="%.2f",
                    help="Average yearly balance in Euros. Negative balances are valid.",
                )
                default = st.selectbox(
                    "Credit in Default",
                    options=VALID_CATEGORIES["default"],
                    index=VALID_CATEGORIES["default"].index(st.session_state["default"]),
                    format_func=lambda x: "Yes" if x == "yes" else "No",
                    help="Does the customer currently have credit in default?",
                )

            with col_f2:
                housing = st.selectbox(
                    "Housing Loan",
                    options=VALID_CATEGORIES["housing"],
                    index=VALID_CATEGORIES["housing"].index(st.session_state["housing"]),
                    format_func=lambda x: "Yes" if x == "yes" else "No",
                    help="Does the customer currently have a housing loan?",
                )
                loan = st.selectbox(
                    "Personal Loan",
                    options=VALID_CATEGORIES["loan"],
                    index=VALID_CATEGORIES["loan"].index(st.session_state["loan"]),
                    format_func=lambda x: "Yes" if x == "yes" else "No",
                    help="Does the customer currently have a personal loan?",
                )

        # Section 3: Campaign Information
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-header">
                    <span class="section-num">3</span>
                    <span class="section-heading-text">Campaign Information</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_c1, col_c2 = st.columns(2)

            with col_c1:
                contact = st.selectbox(
                    "Contact Method",
                    options=VALID_CATEGORIES["contact"],
                    index=VALID_CATEGORIES["contact"].index(st.session_state["contact"]),
                    format_func=lambda x: x.title(),
                    help="Communication type used to reach the customer.",
                )
                month = st.selectbox(
                    "Campaign Month",
                    options=VALID_CATEGORIES["month"],
                    index=VALID_CATEGORIES["month"].index(st.session_state["month"]),
                    format_func=lambda x: x.upper(),
                    help="Month in which contact is initiated.",
                )
                day = st.number_input(
                    "Campaign Day of Month",
                    min_value=1,
                    max_value=31,
                    value=int(st.session_state["day"]),
                    step=1,
                    help="Day of month (1 to 31).",
                )

            with col_c2:
                campaign = st.number_input(
                    "Contacts During Current Campaign",
                    min_value=1,
                    max_value=65,
                    value=int(st.session_state["campaign"]),
                    step=1,
                    help="Number of contacts performed during this campaign.",
                )

                # Prior Contact Interaction
                prior_contact = st.radio(
                    "Was this customer contacted in a prior campaign?",
                    options=["No", "Yes"],
                    index=0 if st.session_state["has_prior_contact"] == "No" else 1,
                    horizontal=True,
                    help="Select No if customer was never contacted previously. Sets pdays = -1 automatically.",
                )

                if prior_contact == "Yes":
                    pdays = st.number_input(
                        "Days Since Previous Contact (pdays)",
                        min_value=0,
                        max_value=900,
                        value=int(st.session_state["pdays_val"]),
                        step=1,
                        help="Number of days since the customer was last contacted from an earlier campaign.",
                    )
                    previous = st.number_input(
                        "Number of Previous Contacts",
                        min_value=1,
                        max_value=100,
                        value=int(st.session_state["previous_val"]),
                        step=1,
                        help="Total contacts performed before this campaign.",
                    )
                    poutcome = st.selectbox(
                        "Outcome of Previous Campaign",
                        options=VALID_CATEGORIES["poutcome"],
                        index=VALID_CATEGORIES["poutcome"].index(st.session_state["poutcome_val"]),
                        format_func=lambda x: x.title(),
                        help="Result of the prior marketing contact.",
                    )
                else:
                    pdays = -1
                    previous = 0
                    poutcome = "unknown"
                    st.markdown(
                        """
                        <div class="custom-info-callout">
                            ℹ️ <strong>First-time contact</strong>: <code>pdays = -1</code> per pipeline specification.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Form Submission Actions
        btn_col1, btn_col2 = st.columns([3, 2])
        with btn_col1:
            submit_btn = st.form_submit_button(
                "Predict Subscription",
                type="primary",
                use_container_width=True,
            )
        with btn_col2:
            reset_btn = st.form_submit_button(
                "Reset Form",
                type="secondary",
                use_container_width=True,
            )

    # Process Form Actions inside Left Column
    if reset_btn:
        do_reset()
        st.rerun()

    if submit_btn:
        if not model_loaded:
            st.error("Prediction cannot be performed: ML model pipeline is not available.")
        else:
            customer_record = {
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "balance": balance,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "day": day,
                "month": month,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "poutcome": poutcome,
            }

            try:
                # Build inference DataFrame (duration strictly excluded; previously_contacted computed)
                input_df = build_inference_dataframe(customer_record)

                # Generate prediction
                result = predict_subscription(model, input_df)
                st.session_state["prediction_result"] = result

                # Update session state to preserve inputs
                st.session_state["age"] = age
                st.session_state["job"] = job
                st.session_state["marital"] = marital
                st.session_state["education"] = education
                st.session_state["balance"] = balance
                st.session_state["default"] = default
                st.session_state["housing"] = housing
                st.session_state["loan"] = loan
                st.session_state["contact"] = contact
                st.session_state["month"] = month
                st.session_state["day"] = day
                st.session_state["campaign"] = campaign
                st.session_state["has_prior_contact"] = prior_contact
                if prior_contact == "Yes":
                    st.session_state["pdays_val"] = pdays
                    st.session_state["previous_val"] = previous
                    st.session_state["poutcome_val"] = poutcome

            except Exception as exc:
                st.error(f"Prediction Error: {exc}")


with col_panel:
    # -------------------------------------------------------------------------
    # Prediction Results Panel (Right Column, Aligned Near Top)
    # -------------------------------------------------------------------------
    pred_res = st.session_state.get("prediction_result")

    if pred_res is None:
        # Before First Prediction: Attractive Clean Empty State
        st.markdown(
            """
            <div class="results-panel-box empty-state-card">
                <div class="empty-icon-wrap">
                    <span style="font-size: 26px;">📊</span>
                </div>
                <h3 class="empty-heading">Prediction Results</h3>
                <p class="empty-text">
                    Enter customer details and run a prediction to see the estimated subscription likelihood.
                </p>
                <div style="display: inline-block;">
                    <span class="model-indicator">
                        <span class="model-dot"></span> Random Forest · Tuned Pipeline
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # After Prediction: Polished Results Card
        is_sub = pred_res["is_subscriber"]
        prob_pct = pred_res["positive_probability"] * 100

        card_class = "result-box-positive" if is_sub else "result-box-neutral"
        badge_class = "result-badge-positive" if is_sub else "result-badge-neutral"
        badge_text = "Positive Outlook" if is_sub else "Standard Outlook"
        text_class = "positive-text" if is_sub else "neutral-text"
        meter_fill_class = "meter-fill-positive" if is_sub else "meter-fill-neutral"

        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="result-title-row">
                    <span class="{badge_class}">
                        <span>{'✓' if is_sub else '•'}</span> {badge_text}
                    </span>
                    <span style="font-size: 12px; color: #64748B; font-weight: 500;">
                        {pred_res['model_name']}
                    </span>
                </div>
                <h2 class="result-outcome-text {text_class}">
                    {pred_res['prediction_text']}
                </h2>
                <p class="result-detail-text">
                    Estimated probability of subscription: <strong>{prob_pct:.1f}%</strong>
                </p>
                <div class="meter-container">
                    <div class="meter-track">
                        <div class="{meter_fill_class}" style="width: {min(max(prob_pct, 1.0), 100.0):.1f}%;"></div>
                    </div>
                    <div class="meter-labels">
                        <span>0%</span>
                        <span style="color: #64748B;">50% Baseline</span>
                        <span>100%</span>
                    </div>
                </div>
                <div class="result-footer-disclaimer">
                    {pred_res['disclaimer']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Secondary Model Specifications Card (Right Column Below Results)
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="model-info-card">
            <div class="info-card-title">Model Specifications</div>
            <div class="info-row">
                <span class="info-label">Model</span>
                <span class="info-val">Random Forest (Tuned)</span>
            </div>
            <div class="info-row">
                <span class="info-label">Prediction Type</span>
                <span class="info-val">Binary Classification</span>
            </div>
            <div class="info-row">
                <span class="info-label">Target</span>
                <span class="info-val">Term Deposit (yes / no)</span>
            </div>
            <div class="info-row">
                <span class="info-label">Timing Constraint</span>
                <span class="info-val">Pre-contact (duration excluded)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

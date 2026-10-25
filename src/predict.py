"""
Inference and Prediction Module for Bank Marketing Predictor.

Handles:
- Loading and caching the serialized scikit-learn pipeline.
- Input data validation.
- Schema construction matching exact training features.
- Generating predictions and positive-class ('yes') subscription probabilities.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"

EXPECTED_FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "previously_contacted",
]

VALID_CATEGORIES = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["failure", "other", "success", "unknown"],
}


def load_trained_model(model_path: Optional[Path] = None) -> Any:
    """
    Load the saved Random Forest pipeline from disk.

    Raises:
        FileNotFoundError: If the model file is not present.
        RuntimeError: If loading or unpickling fails.
    """
    path = model_path or DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {path}\n"
            "Please generate the model artifact first by running: python export_model.py"
        )
    try:
        pipeline = joblib.load(path)
        return pipeline
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc


def validate_customer_input(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate input ranges and categorical constraints.

    Returns:
        (is_valid, error_message)
    """
    # Numerical validation
    if not isinstance(data.get("age"), (int, float)) or not (18 <= data["age"] <= 120):
        return False, "Age must be an integer between 18 and 120."

    if not isinstance(data.get("balance"), (int, float)):
        return False, "Balance must be a valid numerical value."

    if not isinstance(data.get("day"), (int, float)) or not (1 <= data["day"] <= 31):
        return False, "Contact day must be between 1 and 31."

    if not isinstance(data.get("campaign"), (int, float)) or data["campaign"] < 1:
        return False, "Campaign contacts must be at least 1."

    # pdays validation (-1 or >= 0)
    pdays = data.get("pdays")
    if not isinstance(pdays, (int, float)) or (pdays < 0 and pdays != -1):
        return False, "Days since previous contact (pdays) must be -1 (not contacted) or >= 0."

    # previous validation (>= 0)
    previous = data.get("previous")
    if not isinstance(previous, (int, float)) or previous < 0:
        return False, "Number of previous contacts must be >= 0."

    # Categorical validation
    for cat_col, valid_vals in VALID_CATEGORIES.items():
        val = data.get(cat_col)
        if val not in valid_vals:
            return False, f"Invalid value for '{cat_col}': '{val}'. Must be one of {valid_vals}"

    return True, None


def build_inference_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Construct a single-row DataFrame with the exact 16 expected features and types.
    Ensures 'previously_contacted' indicator is automatically computed from pdays.
    Excludes 'duration' completely.
    """
    is_valid, error_msg = validate_customer_input(data)
    if not is_valid:
        raise ValueError(f"Input validation error: {error_msg}")

    # Compute previously_contacted feature
    pdays = int(data["pdays"])
    previously_contacted = 1 if pdays != -1 else 0

    record = {
        "age": int(data["age"]),
        "job": str(data["job"]),
        "marital": str(data["marital"]),
        "education": str(data["education"]),
        "default": str(data["default"]),
        "balance": float(data["balance"]),
        "housing": str(data["housing"]),
        "loan": str(data["loan"]),
        "contact": str(data["contact"]),
        "day": int(data["day"]),
        "month": str(data["month"]),
        "campaign": int(data["campaign"]),
        "pdays": pdays,
        "previous": int(data["previous"]),
        "poutcome": str(data["poutcome"]),
        "previously_contacted": previously_contacted,
    }

    df = pd.DataFrame([record])
    # Ensure column order strictly matches EXPECTED_FEATURE_COLUMNS
    return df[EXPECTED_FEATURE_COLUMNS]


def predict_subscription(model: Any, input_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run prediction on a validated DataFrame using the loaded pipeline.

    Returns:
        Dict with prediction outcome, probability, model name, and disclaimer.
    """
    # Verify classes
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)

    if "yes" not in classes:
        raise ValueError(f"Expected positive class 'yes' in classifier classes: {classes}")

    pos_index = classes.index("yes")

    raw_prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    positive_prob = float(probabilities[pos_index])

    is_subscriber = raw_prediction == "yes"

    return {
        "prediction_text": "Likely to subscribe" if is_subscriber else "Unlikely to subscribe",
        "raw_prediction": raw_prediction,
        "is_subscriber": is_subscriber,
        "positive_probability": positive_prob,
        "positive_class": "yes",
        "model_name": "Random Forest (Tuned Pipeline)",
        "disclaimer": "Predictions are estimates based on historical data and do not guarantee actual customer behavior.",
    }

"""
Common configuration, shared helper functions, and pipeline builders.

This module centralizes reusable logic from `notebooks/bank_marketing_ml.ipynb`
so that section-specific scripts can import and reuse standard dataset loading,
feature engineering, preprocessing, and model definitions without code duplication.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------------------------
# Paths and Constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bank-full.csv"

TARGET = "y"

ORIGINAL_NUMERICAL_FEATURES = [
    "age", "balance", "day", "duration",
    "campaign", "pdays", "previous"
]

ORIGINAL_CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

# Model input features (excludes duration because it is unavailable pre-contact)
MODEL_NUMERICAL_FEATURES = [
    "age", "balance", "day", "campaign", "pdays", "previous", "previously_contacted"
]

MODEL_CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Data Loading and Feature Engineering
# -----------------------------------------------------------------------------

def load_raw_dataset(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the primary bank marketing dataset from CSV.
    The file uses semicolon (;) delimiter.
    """
    path = data_path or DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {path}")
    return pd.read_csv(path, sep=";")


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Section 7 feature engineering:
    Creates 'previously_contacted' binary indicator (0 if pdays == -1 else 1).
    Does not modify the input DataFrame in-place.
    """
    df_out = df.copy()
    df_out["previously_contacted"] = (df_out["pdays"] != -1).astype(int)
    return df_out


def get_feature_target_split(df: pd.DataFrame, target_col: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and target series y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform 80:20 stratified train-test split (Section 8).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def prepare_model_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop 'duration' column from train and test inputs (Section 9.1).
    """
    X_train_model = X_train.drop(columns=["duration"], errors="ignore")
    X_test_model = X_test.drop(columns=["duration"], errors="ignore")
    return X_train_model, X_test_model


# -----------------------------------------------------------------------------
# Preprocessing and Pipeline Creation
# -----------------------------------------------------------------------------

def create_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Create a ColumnTransformer with StandardScaler for numerical features
    and OneHotEncoder(handle_unknown='ignore') for categorical features (Section 9.3 & 10.2).
    """
    num_cols = numerical_features or MODEL_NUMERICAL_FEATURES
    cat_cols = categorical_features or MODEL_CATEGORICAL_FEATURES

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )


def build_baseline_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Instantiate the four baseline classifiers (Section 10.3).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        )
    }


def build_model_pipelines(
    models: Optional[Dict[str, Any]] = None,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Dict[str, Pipeline]:
    """
    Combine preprocessor and classifier into unified pipelines (Section 10.4).
    """
    model_dict = models or build_baseline_models()
    pipelines = {}
    for name, model in model_dict.items():
        preprocessor = create_preprocessor(numerical_features, categorical_features)
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clone(model))
            ]
        )
    return pipelines


# -----------------------------------------------------------------------------
# Cross-Validation & Hyperparameter Tuning Configuration
# -----------------------------------------------------------------------------

def get_scoring_metrics() -> Dict[str, Any]:
    """
    Define the evaluation scoring dictionary used during hyperparameter search (Section 13.2 / 13.4).
    """
    return {
        "f1": make_scorer(
            f1_score,
            pos_label="yes",
            zero_division=0
        ),
        "precision": make_scorer(
            precision_score,
            pos_label="yes",
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            pos_label="yes",
            zero_division=0
        ),
        "roc_auc": "roc_auc"
    }


def get_cv_strategy(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """
    Define the 5-fold stratified cross-validation strategy (Section 13.2).
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def get_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """
    Define candidate hyperparameter search spaces (Section 13.3).
    """
    return {
        "Logistic Regression": {
            "classifier__C": [0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 15],
            "classifier__min_samples_leaf": [1, 3],
            "classifier__class_weight": [None, "balanced"]
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3]
        }
    }

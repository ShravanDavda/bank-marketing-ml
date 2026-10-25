"""
Section 9: Data Preprocessing
Corresponds to Cells 96-109 of `notebooks/bank_marketing_ml.ipynb`.

Builds and verifies the data preprocessing pipeline:
- Excludes 'duration' from model input features (pre-contact prediction constraint).
- Standardizes numerical features using StandardScaler.
- One-hot encodes categorical features using OneHotEncoder(handle_unknown='ignore').
- Fits preprocessor strictly on training data and transforms test data.
- Verifies output matrix finiteness and dimension alignment.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    TARGET,
    apply_feature_engineering,
    create_preprocessor,
    get_feature_target_split,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)


def has_only_finite_values(matrix) -> bool:
    """
    Check if a dense array or sparse matrix contains only finite numerical values (Section 9.6).
    """
    if sparse.issparse(matrix):
        return bool(np.isfinite(matrix.data).all())
    return bool(np.isfinite(matrix).all())


def run_preprocessing(
    X_train_model: pd.DataFrame,
    X_test_model: pd.DataFrame
) -> Tuple[ColumnTransformer, any, any]:
    """
    Build and fit the ColumnTransformer on training features and transform test features.
    """
    preprocessor = create_preprocessor(
        MODEL_NUMERICAL_FEATURES,
        MODEL_CATEGORICAL_FEATURES
    )
    X_train_processed = preprocessor.fit_transform(X_train_model)
    X_test_processed = preprocessor.transform(X_test_model)

    return preprocessor, X_train_processed, X_test_processed


def verify_preprocessing_output(
    X_train_processed: any,
    X_test_processed: any,
    X_train_model: pd.DataFrame,
    X_test_model: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> None:
    """
    Execute Section 9.6 assertions.
    """
    assert has_only_finite_values(X_train_processed), "Non-finite values found in X_train_processed."
    assert has_only_finite_values(X_test_processed), "Non-finite values found in X_test_processed."
    assert X_train_processed.shape[0] == len(y_train), "Row count mismatch with y_train."
    assert X_test_processed.shape[0] == len(y_test), "Row count mismatch with y_test."
    assert X_train_model.columns.equals(X_test_model.columns), "Feature column mismatch between train and test."


def main() -> None:
    print("=" * 70)
    print("Section 9: Data Preprocessing")
    print("=" * 70)

    # Prepare data
    df_raw = load_raw_dataset()
    df = apply_feature_engineering(df_raw)
    X, y = get_feature_target_split(df, TARGET)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 9.1 Model Input Features (duration excluded)
    X_train_model, X_test_model = prepare_model_inputs(X_train, X_test)
    print("\n9.1 Model Input Features:")
    print(f"  - Training input shape (duration excluded): {X_train_model.shape}")
    print(f"  - Test input shape (duration excluded)    : {X_test_model.shape}")

    # 9.2 Identify Feature Types
    print("\n9.2 Feature Groups:")
    print(f"  - Numerical features ({len(MODEL_NUMERICAL_FEATURES)}): {MODEL_NUMERICAL_FEATURES}")
    print(f"  - Categorical features ({len(MODEL_CATEGORICAL_FEATURES)}): {MODEL_CATEGORICAL_FEATURES}")

    # 9.3 & 9.4 Preprocessing Pipeline & Fit
    preprocessor, X_train_proc, X_test_proc = run_preprocessing(X_train_model, X_test_model)
    print("\n9.3 & 9.4 Preprocessing Applied:")
    print(f"  - Original training shape: {X_train_model.shape}")
    print(f"  - Processed training shape: {X_train_proc.shape}")
    print(f"  - Original test shape: {X_test_model.shape}")
    print(f"  - Processed test shape: {X_test_proc.shape}")

    # 9.6 Verification
    verify_preprocessing_output(
        X_train_proc, X_test_proc, X_train_model, X_test_model, y_train, y_test
    )
    print("\n9.6 Verification Passed:")
    print("  - All processed values are finite.")
    print("  - Row dimensions strictly match target series.")
    print("  - Preprocessor fitted strictly on training data without leakage.")


if __name__ == "__main__":
    main()

"""
Section 8: Train-Test Split
Corresponds to Cells 83-95 of `notebooks/bank_marketing_ml.ipynb`.

Separates features and target variable, then performs an 80:20 stratified split
with random_state=42. Verifies subset sizes, index disjunction, and target distribution.
"""

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    RANDOM_STATE,
    TARGET,
    apply_feature_engineering,
    get_feature_target_split,
    load_raw_dataset,
    split_train_test,
)


def perform_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split DataFrame into X_train, X_test, y_train, y_test using stratified sampling.
    """
    X, y = get_feature_target_split(df, TARGET)
    return split_train_test(X, y, test_size=test_size, random_state=random_state)


def verify_split_integrity(
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> None:
    """
    Verify Section 8.5 assertions:
    - Partition completeness
    - Non-overlapping indices
    - Alignment of feature indices with target indices
    """
    assert len(X_train) + len(X_test) == len(X), "Row count mismatch after split."
    assert X_train.index.intersection(X_test.index).empty, "Train and test indices overlap."
    assert X_train.index.equals(y_train.index), "X_train and y_train indices misaligned."
    assert X_test.index.equals(y_test.index), "X_test and y_test indices misaligned."


def main() -> None:
    print("=" * 70)
    print("Section 8: Train-Test Split")
    print("=" * 70)

    df_raw = load_raw_dataset()
    df = apply_feature_engineering(df_raw)

    # 8.1 Separate Features and Target
    X, y = get_feature_target_split(df, TARGET)
    print(f"\n8.1 Features shape: {X.shape}, Target shape: {y.shape}")

    # 8.2 Split the Dataset
    X_train, X_test, y_train, y_test = perform_train_test_split(df)

    # 8.3 Inspect Dimensions
    print("\n8.3 Split Dimensions:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test : {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test : {y_test.shape}")

    # 8.4 Target Distribution
    print("\n8.4 Target Class Proportions:")
    train_dist = (y_train.value_counts(normalize=True) * 100).round(2)
    test_dist = (y_test.value_counts(normalize=True) * 100).round(2)
    print("  Training subset (%):\n", train_dist)
    print("  Test subset (%):\n", test_dist)

    # 8.5 Verification
    verify_split_integrity(X, X_train, X_test, y_train, y_test)
    print("\n8.5 Verification Passed:")
    print("  - Split accounts for 100% of records.")
    print("  - Indices are strictly disjoint.")
    print("  - Features and labels remain perfectly aligned.")


if __name__ == "__main__":
    main()

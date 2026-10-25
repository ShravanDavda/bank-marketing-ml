"""
Section 4: Data Understanding
Corresponds to Cells 5-33 of `notebooks/bank_marketing_ml.ipynb`.

Performs initial structure, variable, data type, missing values, duplicates,
and distribution checks on the Bank Marketing dataset.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    ORIGINAL_CATEGORICAL_FEATURES,
    ORIGINAL_NUMERICAL_FEATURES,
    TARGET,
    load_raw_dataset,
)


def get_feature_groups() -> Tuple[List[str], List[str], str]:
    """
    Return numerical features, categorical features, and target column name (Section 4.4).
    """
    return (
        ORIGINAL_NUMERICAL_FEATURES.copy(),
        ORIGINAL_CATEGORICAL_FEATURES.copy(),
        TARGET
    )


def summarize_target_distribution(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """
    Compute count and percentage distribution for the target variable (Section 4.7).
    """
    counts = df[target_col].value_counts()
    percentages = (df[target_col].value_counts(normalize=True) * 100).round(2)
    return pd.DataFrame({"Count": counts, "Percentage (%)": percentages})


def run_data_understanding(df: pd.DataFrame) -> Dict[str, any]:
    """
    Execute all understanding checks from Section 4.
    """
    num_features, cat_features, target = get_feature_groups()

    results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "numerical_features": num_features,
        "categorical_features": cat_features,
        "target": target,
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": summarize_target_distribution(df, target),
        "unique_counts": df.nunique().to_dict(),
        "pdays_negative_one_count": int((df["pdays"] == -1).sum())
    }
    return results


def main() -> None:
    print("=" * 70)
    print("Section 4: Data Understanding")
    print("=" * 70)

    df = load_raw_dataset()

    # 4.1 Dataset Shape
    print(f"\n4.1 Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 4.2 Dataset Info
    print("\n4.2 Dataset Information (summary):")
    print(df.dtypes.value_counts())

    # 4.3 Column Names
    print(f"\n4.3 Column Names ({len(df.columns)}):")
    print(df.columns.tolist())

    # 4.4 Feature Groups
    num_cols, cat_cols, target = get_feature_groups()
    print(f"\n4.4 Feature Classification:")
    print(f"  - Numerical features ({len(num_cols)}): {num_cols}")
    print(f"  - Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"  - Target: {target}")

    # 4.5 Missing Values
    total_missing = df.isnull().sum().sum()
    print(f"\n4.5 Missing Values: Total = {total_missing}")

    # 4.6 Duplicate Records
    total_dupes = df.duplicated().sum()
    print(f"\n4.6 Duplicate Rows: Total = {total_dupes}")

    # 4.7 Target Distribution
    print("\n4.7 Target Distribution:")
    target_dist = summarize_target_distribution(df, target)
    print(target_dist)

    # 4.8 Descriptive Statistics
    print("\n4.8 Descriptive Statistics (selected numerical):")
    print(df[num_cols].describe().round(2))

    # 4.9 Unique Values
    print("\n4.9 Unique Values Count:")
    print(df.nunique())

    # 4.10 Key Observations Summary
    print("\n4.10 Key Observations:")
    print("  1. Primary dataset contains 45,211 records and 17 columns (16 features + target 'y').")
    print("  2. No missing values or completely duplicated rows are present.")
    print("  3. Target class is imbalanced (~88.3% 'no' vs ~11.7% 'yes').")
    print("  4. 'pdays' uses -1 to represent customers not previously contacted (36,954 records).")


if __name__ == "__main__":
    main()

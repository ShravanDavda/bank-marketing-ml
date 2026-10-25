"""
Section 5: Data Cleaning
Corresponds to Cells 34-52 of `notebooks/bank_marketing_ml.ipynb`.

Audits the dataset for data quality issues: missing values, duplicate records,
numerical boundaries, categorical values (including 'unknown'), and text whitespace.
Confirms cleaning decisions: no records removed; preserve 'pdays = -1' and negative balances.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

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


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform complete data quality audit as implemented in Section 5.
    """
    original_shape = df.shape
    total_missing = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())

    # Numerical conditions (Section 5.3)
    num_checks = {
        "age_below_18": int((df["age"] < 18).sum()),
        "negative_balance": int((df["balance"] < 0).sum()),
        "negative_duration": int((df["duration"] < 0).sum()),
        "negative_campaign": int((df["campaign"] < 0).sum()),
        "pdays_equal_negative_one": int((df["pdays"] == -1).sum()),
    }

    # Text whitespace checks (Section 5.4)
    whitespace_issues = {}
    for column in ORIGINAL_CATEGORICAL_FEATURES + [TARGET]:
        values = df[column].dropna().astype(str)
        ws_count = int(values.ne(values.str.strip()).sum())
        whitespace_issues[column] = ws_count

    return {
        "original_shape": original_shape,
        "missing_values": total_missing,
        "duplicate_rows": total_duplicates,
        "numerical_checks": num_checks,
        "whitespace_issues": whitespace_issues,
    }


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and return the cleaned dataset.
    Following Section 5 decisions: no records are dropped because audit established
    0 nulls, 0 duplicates, 0 whitespace issues, and valid negative balances / pdays = -1.
    """
    audit = audit_data_quality(df)
    df_cleaned = df.copy()

    # Verification (Section 5.6)
    assert df_cleaned.shape == audit["original_shape"]
    assert df_cleaned.isnull().sum().sum() == 0
    assert df_cleaned.duplicated().sum() == 0

    return df_cleaned, audit


def main() -> None:
    print("=" * 70)
    print("Section 5: Data Cleaning & Quality Audit")
    print("=" * 70)

    df = load_raw_dataset()
    df_clean, audit = clean_dataset(df)

    print("\n5.1 Data Quality Audit:")
    print(f"  - Original shape: {audit['original_shape']}")
    print(f"  - Missing values: {audit['missing_values']}")
    print(f"  - Duplicate rows: {audit['duplicate_rows']}")

    print("\n5.2 Categorical Value Inspection:")
    for col in ORIGINAL_CATEGORICAL_FEATURES:
        unique_vals = df_clean[col].unique().tolist()
        unknown_count = (df_clean[col] == "unknown").sum()
        print(f"  - {col:<12}: {len(unique_vals)} unique values (contains 'unknown': {unknown_count:,})")

    print("\n5.3 Numerical Range Inspection:")
    for check, count in audit["numerical_checks"].items():
        print(f"  - {check:<28}: {count:,}")

    print("\n5.4 Whitespace Inconsistency Inspection:")
    for col, count in audit["whitespace_issues"].items():
        print(f"  - {col:<12}: {count} values with surrounding whitespace")

    print("\n5.5 Cleaning Decisions:")
    print("  - Missing values: None found; no imputation needed.")
    print("  - Duplicate rows: None found; no row deletion needed.")
    print("  - 'unknown' categories: Retained as valid domain categories.")
    print("  - Negative balance: Retained (negative account balance is valid).")
    print("  - 'pdays = -1': Retained (indicates no prior campaign contact).")

    print("\n5.6 Verification:")
    print(f"  - Dataset shape after audit/cleaning: {df_clean.shape}")
    print(f"  - Verification passed: 100% integrity maintained.")


if __name__ == "__main__":
    main()

"""
Section 7: Feature Engineering
Corresponds to Cells 72-82 of `notebooks/bank_marketing_ml.ipynb`.

Creates and verifies the `previously_contacted` binary indicator derived from `pdays`:
  - 0 when pdays == -1 (customer was not previously contacted)
  - 1 when pdays != -1 (customer had at least one prior contact)
Retains both `pdays` and `previously_contacted` for model inputs.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    apply_feature_engineering,
    load_raw_dataset,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Section 7 feature engineering:
    Creates 'previously_contacted' indicator without modifying the original DataFrame.
    """
    return apply_feature_engineering(df)


def verify_feature_engineering(df_engineered: pd.DataFrame) -> None:
    """
    Verify feature assertions from Section 7.2:
    - Values must be exclusively 0 or 1
    - Values must match (pdays != -1).astype(int) exactly
    """
    assert "previously_contacted" in df_engineered.columns, "Feature 'previously_contacted' missing."
    assert df_engineered["previously_contacted"].isin([0, 1]).all(), "Feature contains non-binary values."
    assert (
        df_engineered["previously_contacted"]
        == (df_engineered["pdays"] != -1).astype(int)
    ).all(), "Feature values do not match pdays condition."


def main() -> None:
    print("=" * 70)
    print("Section 7: Feature Engineering")
    print("=" * 70)

    df_raw = load_raw_dataset()
    print(f"Original dataset shape: {df_raw.shape}")

    # 7.1 Previous Contact Indicator
    df_engineered = engineer_features(df_raw)
    print("\n7.1 Previous Contact Indicator created:")
    print(df_engineered[["pdays", "previously_contacted"]].head(10))

    counts = df_engineered["previously_contacted"].value_counts()
    print(f"\nValue counts of 'previously_contacted':\n{counts}")

    # 7.2 Feature Verification
    verify_feature_engineering(df_engineered)
    print("\n7.2 Feature Verification Passed:")
    print("  - Binary constraint verified (all values in [0, 1]).")
    print("  - Consistency with pdays != -1 verified.")
    print(f"  - Dataset shape after feature engineering: {df_engineered.shape}")

    # 7.3 Decisions
    print("\n7.3 Feature Engineering Decisions:")
    print("  - Retained both 'pdays' and 'previously_contacted'.")
    print("  - Duration will be excluded in the preprocessing stage.")


if __name__ == "__main__":
    main()

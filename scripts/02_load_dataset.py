"""
Section 3: Load Dataset
Corresponds to Cells 3-4 of `notebooks/bank_marketing_ml.ipynb`.

The Bank Marketing dataset contains customer, financial, and marketing campaign
information. We use `bank-full.csv` as our primary dataset because it contains
45,211 records. The file uses semicolon (;) as its field separator.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import DEFAULT_DATA_PATH, load_raw_dataset


def load_dataset(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the primary bank marketing dataset.

    Parameters:
        data_path: Optional path to the CSV file. If None, uses default path.

    Returns:
        pd.DataFrame containing 45,211 records and 17 columns.
    """
    return load_raw_dataset(data_path)


def main() -> None:
    print("=" * 70)
    print("Section 3: Load Dataset")
    print("=" * 70)
    print(f"Loading data from: {DEFAULT_DATA_PATH}")

    df = load_dataset()

    print(f"\nDataset loaded successfully.")
    print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("\nFirst 5 records:")
    print(df.head())


if __name__ == "__main__":
    main()

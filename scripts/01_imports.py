"""
Section 1: Problem Definition & Section 2: Import Libraries
Corresponds to Cells 0-2 of `notebooks/bank_marketing_ml.ipynb`.

Problem Statement:
    Predict whether a bank customer will subscribe to a term deposit based on customer
    and marketing campaign information.

Machine Learning Objective:
    Supervised binary classification where target `y` indicates whether the customer
    subscribed to a term deposit ('yes' or 'no').

Business Objective:
    Identify customers more likely to subscribe to a term deposit, potentially enabling
    more targeted, cost-effective marketing campaigns.
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_environment() -> dict:
    """
    Verify required libraries and environment versions for the project.
    """
    dependencies = {}

    import numpy as np
    dependencies["numpy"] = np.__version__

    import pandas as pd
    dependencies["pandas"] = pd.__version__

    import matplotlib
    dependencies["matplotlib"] = matplotlib.__version__

    import seaborn as sns
    dependencies["seaborn"] = sns.__version__

    import sklearn
    dependencies["scikit-learn"] = sklearn.__version__

    import scipy
    dependencies["scipy"] = scipy.__version__

    try:
        import joblib
        dependencies["joblib"] = joblib.__version__
    except ImportError:
        dependencies["joblib"] = "Not installed"

    return dependencies


def main() -> None:
    print("=" * 70)
    print("Bank Marketing ML - Problem Definition & Environment Verification")
    print("=" * 70)
    print("\nProject Objectives:")
    print("  - Problem Type: Binary Classification (Predict term deposit subscription)")
    print("  - Target Variable: 'y' ('yes' / 'no')")
    print("  - Primary Dataset: data/bank-full.csv (45,211 rows, 17 columns)")
    print("  - Key Constraint: Contact duration is excluded from model inputs")
    print("    (intended for pre-contact decision making).")

    print("\nVerifying Core Project Dependencies...")
    versions = check_environment()
    for lib, ver in versions.items():
        print(f"  - {lib:<15}: {ver}")

    print("\nEnvironment verification completed successfully.")


if __name__ == "__main__":
    main()

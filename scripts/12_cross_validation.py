"""
Section 13 (Part 1): Cross-Validation Strategy & Search Spaces
Corresponds to Cells 152-158 of `notebooks/bank_marketing_ml.ipynb`.

Configures the cross-validation and hyperparameter tuning setup:
- 13.1 Cross-validation tools and metric scorers.
- 13.2 Five-fold stratified cross-validation on training data (StratifiedKFold).
       Primary metric: F1-Score (pos_label='yes', zero_division=0).
       Secondary metrics: Precision, Recall, and ROC-AUC.
- 13.3 Hyperparameter search spaces for:
       - Logistic Regression (6 combinations)
       - Random Forest (16 combinations)
       - Gradient Boosting (8 combinations)
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

from sklearn.model_selection import StratifiedKFold

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    get_cv_strategy,
    get_param_grids,
    get_scoring_metrics,
)


def get_cross_validation_strategy(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """
    Define 5-fold stratified cross-validation (Section 13.2).
    """
    return get_cv_strategy(n_splits=n_splits, random_state=random_state)


def get_tuning_scoring_metrics() -> Dict[str, Any]:
    """
    Define the scoring metrics dictionary with F1 as primary scorer (Section 13.2).
    """
    return get_scoring_metrics()


def get_candidate_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """
    Define hyperparameter grids for the 3 candidate models (Section 13.3).
    """
    return get_param_grids()


def calculate_grid_combinations(param_grids: Dict[str, Dict[str, List[Any]]]) -> Dict[str, int]:
    """
    Calculate number of hyperparameter combinations for each candidate grid.
    """
    counts = {}
    for name, grid in param_grids.items():
        combinations = 1
        for values in grid.values():
            combinations *= len(values)
        counts[name] = combinations
    return counts


def main() -> None:
    print("=" * 70)
    print("Section 12: Cross-Validation & Hyperparameter Search Spaces")
    print("=" * 70)

    # 13.2 CV Strategy
    cv = get_cross_validation_strategy()
    print("\n13.2 Cross-Validation Strategy:")
    print(f"  - Strategy: StratifiedKFold (n_splits={cv.n_splits}, shuffle={cv.shuffle}, random_state={cv.random_state})")
    print("  - Target-class proportions preserved in every fold.")
    print("  - Preprocessing will be fitted strictly within training folds.")

    # 13.2 Scoring Metrics
    scorers = get_tuning_scoring_metrics()
    print("\n13.2 Evaluation Scorers:")
    print("  - Primary Metric: F1-Score (pos_label='yes', zero_division=0)")
    print("  - Secondary Metrics: Precision, Recall, ROC-AUC")

    # 13.3 Hyperparameter Search Spaces
    param_grids = get_candidate_param_grids()
    combinations = calculate_grid_combinations(param_grids)

    print("\n13.3 Candidate Hyperparameter Search Spaces:")
    for name, grid in param_grids.items():
        print(f"\n  {name} ({combinations[name]} combinations):")
        for param, vals in grid.items():
            print(f"    - {param}: {vals}")


if __name__ == "__main__":
    main()

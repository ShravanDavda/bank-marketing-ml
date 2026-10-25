"""
Section 13 (Part 2): Hyperparameter Search and Tuned Pipelines
Corresponds to Cells 159-168 of `notebooks/bank_marketing_ml.ipynb`.

Executes GridSearchCV across candidate models:
- Evaluates specified hyperparameter configurations using 5-fold stratified CV on training data.
- Optimizes primarily for F1-Score (pos_label='yes').
- Automatically refits the best configuration on the complete training set (refit='f1').
- Extracts best cross-validation scores, best parameters, and fitted tuned pipelines.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    TARGET,
    apply_feature_engineering,
    build_baseline_models,
    build_model_pipelines,
    get_cv_strategy,
    get_feature_target_split,
    get_param_grids,
    get_scoring_metrics,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)


def run_hyperparameter_search(
    pipelines: Dict[str, Pipeline],
    param_grids: Dict[str, Dict[str, List[Any]]],
    X_train_model: pd.DataFrame,
    y_train: pd.Series,
    candidate_names: Optional[List[str]] = None,
    n_jobs: int = 1
) -> Tuple[Dict[str, GridSearchCV], pd.DataFrame]:
    """
    Execute GridSearchCV across specified candidate models (Section 13.4).
    """
    cv_strategy = get_cv_strategy()
    scoring_metrics = get_scoring_metrics()

    selected_candidates = candidate_names or list(param_grids.keys())
    tuning_results = {}
    tuning_summaries = []

    for name in selected_candidates:
        if name not in pipelines or name not in param_grids:
            continue

        print(f"\nTuning {name}...")
        search = GridSearchCV(
            estimator=clone(pipelines[name]),
            param_grid=param_grids[name],
            scoring=scoring_metrics,
            refit="f1",
            cv=cv_strategy,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score="raise"
        )

        start_time = time.perf_counter()
        search.fit(X_train_model, y_train)
        elapsed_time = time.perf_counter() - start_time

        tuning_results[name] = search
        best_index = search.best_index_
        results = search.cv_results_

        tuning_summaries.append({
            "Model": name,
            "Best CV F1": results["mean_test_f1"][best_index],
            "CV Precision": results["mean_test_precision"][best_index],
            "CV Recall": results["mean_test_recall"][best_index],
            "CV ROC-AUC": results["mean_test_roc_auc"][best_index],
            "Search Time (seconds)": round(elapsed_time, 2),
            "Best Parameters": search.best_params_
        })

        print(f"  Best CV F1: {search.best_score_:.4f}")
        print(f"  Best parameters: {search.best_params_}")

    summary_df = pd.DataFrame(tuning_summaries)
    return tuning_results, summary_df


def extract_tuned_pipelines(tuning_results: Dict[str, GridSearchCV]) -> Dict[str, Pipeline]:
    """
    Extract the best refitted pipeline for each candidate (Section 13.7).
    """
    return {
        name: search.best_estimator_
        for name, search in tuning_results.items()
    }


def main() -> None:
    print("=" * 70)
    print("Section 13: Hyperparameter Tuning Execution")
    print("=" * 70)

    # Prepare training data
    df_raw = load_raw_dataset()
    df = apply_feature_engineering(df_raw)
    X, y = get_feature_target_split(df, TARGET)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_model, _ = prepare_model_inputs(X_train, X_test)

    pipelines = build_model_pipelines()
    param_grids = get_param_grids()

    # Note: To avoid excessive runtime when running interactive checks,
    # tuning focuses on the primary candidate Random Forest.
    # To tune all models, pass candidate_names=None.
    target_candidates = ["Random Forest"]
    print(f"\nRunning 5-Fold GridSearch for candidate: {target_candidates}")
    print("Optimization target: F1-Score (pos_label='yes')")

    tuning_results, tuning_summary = run_hyperparameter_search(
        pipelines=pipelines,
        param_grids=param_grids,
        X_train_model=X_train_model,
        y_train=y_train,
        candidate_names=target_candidates,
        n_jobs=-1  # parallelize across available CPU cores for faster execution
    )

    # 13.5 Comparison
    print("\n13.5 Tuning Results Summary:")
    print(tuning_summary.round(4).to_string(index=False))

    # 13.6 Best Parameters
    print("\n13.6 Selected Best Hyperparameters:")
    for name, search in tuning_results.items():
        print(f"\n  {name}:")
        for param, val in search.best_params_.items():
            print(f"    - {param}: {val}")

    # 13.7 Store Tuned Pipelines
    tuned_pipelines = extract_tuned_pipelines(tuning_results)
    print("\n13.7 Tuned Pipelines Stored and Available:")
    for name in tuned_pipelines:
        print(f"  - {name} (refitted on complete training set)")


if __name__ == "__main__":
    main()

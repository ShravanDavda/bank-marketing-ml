"""
Section 10: Model Training
Corresponds to Cells 110-125 of `notebooks/bank_marketing_ml.ipynb`.

Trains four baseline classification models:
- Logistic Regression (max_iter=1000, random_state=42)
- Decision Tree (random_state=42)
- Random Forest (n_estimators=200, random_state=42, n_jobs=-1)
- Gradient Boosting (random_state=42)

Each model is wrapped in a unified pipeline with independent preprocessing.
Excludes 'duration' and fits solely on training data.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
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
    get_feature_target_split,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)


def train_baseline_pipelines(
    X_train_model: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """
    Build and fit all four baseline pipelines, tracking execution time (Section 10.5).
    """
    pipelines = build_model_pipelines()
    training_results = []

    for name, pipeline in pipelines.items():
        start_time = time.perf_counter()
        pipeline.fit(X_train_model, y_train)
        elapsed_time = time.perf_counter() - start_time

        training_results.append({
            "Model": name,
            "Training Time (seconds)": round(elapsed_time, 2),
            "Status": "Trained"
        })

    summary_df = pd.DataFrame(training_results).sort_values(
        by="Training Time (seconds)"
    ).reset_index(drop=True)

    return pipelines, summary_df


def verify_trained_models(pipelines: Dict[str, Pipeline]) -> None:
    """
    Execute Section 10.6 assertions.
    """
    assert len(pipelines) == 4, f"Expected 4 models, got {len(pipelines)}"
    for name, pipeline in pipelines.items():
        classifier = pipeline.named_steps["classifier"]
        assert hasattr(classifier, "classes_"), f"Classifier {name} has not been fitted."


def main() -> None:
    print("=" * 70)
    print("Section 10: Baseline Model Training")
    print("=" * 70)

    # Prepare training split
    df_raw = load_raw_dataset()
    df = apply_feature_engineering(df_raw)
    X, y = get_feature_target_split(df, TARGET)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_model, _ = prepare_model_inputs(X_train, X_test)

    # 10.3 Define Models
    models = build_baseline_models()
    print(f"\n10.3 Defined {len(models)} classification models:")
    for name in models:
        print(f"  - {name}")

    # 10.4 & 10.5 Build and Train Pipelines
    print("\n10.5 Training model pipelines...")
    pipelines, summary_df = train_baseline_pipelines(X_train_model, y_train)

    # 10.6 Verify Trained Models
    verify_trained_models(pipelines)
    print("\n10.6 Verification Passed: All 4 models fitted successfully.")

    # 10.7 Training Results
    print("\n10.7 Training Time Summary:")
    print(summary_df.to_string(index=False))

    print("\n10.8 Summary:")
    print("  - Duration excluded from all model pipelines.")
    print("  - Models are trained and ready for held-out evaluation in Section 11.")


if __name__ == "__main__":
    main()

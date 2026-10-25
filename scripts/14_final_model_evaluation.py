"""
Section 14: Final Model Selection and Evaluation
Corresponds to Cells 169-184 of `notebooks/bank_marketing_ml.ipynb`.

Evaluates the final selected candidate (Tuned Random Forest) on the held-out test dataset:
- Reuses the refitted pipeline from cross-validation tuning.
- Computes final test metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Generates and displays the final confusion matrix.
- Extracts True Negatives, False Positives, False Negatives, and True Positives.
- Compares Baseline Random Forest vs. Tuned Random Forest.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    TARGET,
    apply_feature_engineering,
    build_model_pipelines,
    get_feature_target_split,
    get_param_grids,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)
from scripts.common import RandomForestClassifier


def evaluate_final_candidate(
    final_model: Pipeline,
    X_test_model: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Random Forest"
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate test predictions, evaluate metrics, and construct confusion matrix (Sections 14.2-14.4).
    """
    y_pred = final_model.predict(X_test_model)

    classes = list(final_model.named_steps["classifier"].classes_)
    positive_index = classes.index("yes")
    y_probability = final_model.predict_proba(X_test_model)[:, positive_index]
    y_test_binary = (y_test == "yes").astype(int)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="yes", zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label="yes", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, pos_label="yes", zero_division=0),
        "ROC-AUC": roc_auc_score(y_test_binary, y_probability)
    }

    metrics_df = pd.DataFrame([metrics])
    cm = confusion_matrix(y_test, y_pred, labels=["no", "yes"])

    return metrics_df, cm, y_probability


def compare_baseline_vs_tuned(
    baseline_metrics: Dict[str, float],
    tuned_metrics: Dict[str, float]
) -> pd.DataFrame:
    """
    Construct comparison table between Baseline and Tuned Random Forest (Section 14.5).
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rows = [
        {"Model": "Baseline Random Forest", **{m: baseline_metrics.get(m, 0.0) for m in metrics}},
        {"Model": "Tuned Random Forest", **{m: tuned_metrics.get(m, 0.0) for m in metrics}},
    ]
    return pd.DataFrame(rows)


def plot_final_confusion_matrix(cm: np.ndarray, model_name: str = "Random Forest", show_plot: bool = True) -> None:
    """
    Display the confusion matrix (Section 14.4).
    """
    if not show_plot:
        return

    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    display.plot(colorbar=False, values_format="d")
    plt.title(f"Final Model — {model_name}")
    plt.tight_layout()
    plt.show()


def main() -> None:
    print("=" * 70)
    print("Section 14: Final Model Selection and Evaluation")
    print("=" * 70)

    # Prepare data
    df_raw = load_raw_dataset()
    df = apply_feature_engineering(df_raw)
    X, y = get_feature_target_split(df, TARGET)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_model, X_test_model = prepare_model_inputs(X_train, X_test)

    # 14.1 Select Final Candidate
    # We train the baseline Random Forest and the tuned Random Forest
    print("\n14.1 Selecting candidate: Random Forest...")
    from scripts.common import create_preprocessor
    from sklearn.ensemble import RandomForestClassifier

    # Baseline RF pipeline
    baseline_rf = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    baseline_rf.fit(X_train_model, y_train)
    baseline_eval_df, _, _ = evaluate_final_candidate(baseline_rf, X_test_model, y_test, "Baseline RF")
    baseline_dict = baseline_eval_df.iloc[0].to_dict()

    # Tuned RF pipeline with best hyperparameter settings established by GridSearchCV in the notebook
    # (class_weight='balanced', max_depth=15, min_samples_leaf=3, n_estimators=200)
    tuned_rf = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    tuned_rf.fit(X_train_model, y_train)

    # 14.2 & 14.3 Generate Test Predictions & Metrics
    final_eval_df, final_cm, _ = evaluate_final_candidate(
        tuned_rf, X_test_model, y_test, "Tuned Random Forest"
    )
    tuned_dict = final_eval_df.iloc[0].to_dict()

    print("\n14.3 Final Test Evaluation Metrics:")
    print(final_eval_df.round(4).to_string(index=False))

    # 14.4 Confusion Matrix
    print("\n14.4 Final Confusion Matrix:")
    print(final_cm)
    plot_final_confusion_matrix(final_cm, "Random Forest", show_plot=True)

    # 14.5 Baseline vs. Tuned Comparison
    print("\n14.5 Baseline vs. Tuned Model Comparison:")
    comparison_table = compare_baseline_vs_tuned(baseline_dict, tuned_dict)
    print(comparison_table.round(4).to_string(index=False))

    # 14.6 Confusion Matrix Breakdown
    tn, fp, fn, tp = final_cm.ravel()
    print("\n14.6 Confusion Matrix Breakdown:")
    print(f"  - True Negatives : {tn:,} (Non-subscribers correctly identified)")
    print(f"  - False Positives: {fp:,} (Non-subscribers predicted as subscribers)")
    print(f"  - False Negatives: {fn:,} (Actual subscribers missed)")
    print(f"  - True Positives : {tp:,} (Actual subscribers identified)")


if __name__ == "__main__":
    main()

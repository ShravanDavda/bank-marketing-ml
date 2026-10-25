"""
Section 12: Model Comparison
Corresponds to Cells 142-151 of `notebooks/bank_marketing_ml.ipynb`.

Compares the four baseline classification models across multiple metrics:
- Organizes the comparison table sorted by F1-Score.
- Generates a grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC.
- Identifies the highest-performing model for each individual metric.
- Establishes the shortlist of candidate models for hyperparameter tuning:
  Logistic Regression (baseline), Random Forest (best F1), and Gradient Boosting (best ROC-AUC).
"""

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    TARGET,
    apply_feature_engineering,
    build_model_pipelines,
    get_feature_target_split,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)


def create_comparison_table(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format and sort comparison metrics by F1-Score (Section 12.1).
    """
    metrics = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    return (
        evaluation_df[metrics]
        .sort_values(by="F1-Score", ascending=False)
        .reset_index(drop=True)
    )


def plot_metrics_comparison(comparison_df: pd.DataFrame, show_plot: bool = True) -> None:
    """
    Display grouped bar chart of evaluation metrics by model (Section 12.2).
    """
    if not show_plot:
        return

    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    comparison_df.set_index("Model")[metrics_to_plot].plot(
        kind="bar",
        figsize=(12, 6)
    )
    plt.title("Evaluation Metrics by Model")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Metric")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def identify_strongest_results(comparison_df: pd.DataFrame) -> List[dict]:
    """
    Find which model achieved the highest score for each metric (Section 12.3).
    """
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    strongest = []

    for metric in metrics_to_plot:
        best_index = comparison_df[metric].idxmax()
        best_model = comparison_df.loc[best_index, "Model"]
        best_score = comparison_df.loc[best_index, metric]
        strongest.append({
            "Metric": metric,
            "Best Model": best_model,
            "Score": best_score
        })

    return strongest


def main() -> None:
    print("=" * 70)
    print("Section 12: Model Comparison")
    print("=" * 70)

    # Train and evaluate baseline models
    df_raw = load_raw_dataset()
    df = apply_feature_engineering(df_raw)
    X, y = get_feature_target_split(df, TARGET)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_model, X_test_model = prepare_model_inputs(X_train, X_test)

    # Import evaluation metrics helper from script 10
    from scripts.common import build_model_pipelines
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    pipelines = build_model_pipelines()
    eval_results = []
    y_test_bin = (y_test == "yes").astype(int)

    for name, p in pipelines.items():
        p.fit(X_train_model, y_train)
        pred = p.predict(X_test_model)
        pos_idx = list(p.named_steps["classifier"].classes_).index("yes")
        prob = p.predict_proba(X_test_model)[:, pos_idx]

        eval_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, pos_label="yes", zero_division=0),
            "Recall": recall_score(y_test, pred, pos_label="yes", zero_division=0),
            "F1-Score": f1_score(y_test, pred, pos_label="yes", zero_division=0),
            "ROC-AUC": roc_auc_score(y_test_bin, prob)
        })

    eval_df = pd.DataFrame(eval_results)

    # 12.1 Comparison Table
    print("\n12.1 Model Comparison Table (Ranked by F1-Score):")
    comp_df = create_comparison_table(eval_df)
    print(comp_df.round(4).to_string(index=False))

    # 12.2 Grouped Bar Chart
    print("\n12.2 Plotting metrics comparison...")
    plot_metrics_comparison(comp_df, show_plot=True)

    # 12.3 Strongest Results
    print("\n12.3 Strongest Model per Metric:")
    strongest = identify_strongest_results(comp_df)
    for item in strongest:
        print(f"  - Highest {item['Metric']:<10}: {item['Best Model']} ({item['Score']:.4f})")

    # 12.4 Candidates for Tuning
    candidates = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    print(f"\n12.4 Shortlisted Candidates for Cross-Validation Tuning:")
    for c in candidates:
        print(f"  - {c}")


if __name__ == "__main__":
    main()

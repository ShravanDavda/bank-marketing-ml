"""
Section 11: Model Evaluation
Corresponds to Cells 126-141 of `notebooks/bank_marketing_ml.ipynb`.

Evaluates the four baseline pipelines on the held-out test dataset:
- Generates predictions and positive-class ('yes') subscription probabilities.
- Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Displays confusion matrices, classification reports, and ROC curves.
- Sorts models by F1-Score to reflect performance under class imbalance.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    TARGET,
    apply_feature_engineering,
    get_feature_target_split,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)
from scripts.common import build_model_pipelines


def generate_predictions(
    pipelines: Dict[str, Pipeline],
    X_test_model: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Generate test predictions for each fitted pipeline (Section 11.2).
    """
    predictions = {}
    for name, pipeline in pipelines.items():
        predictions[name] = pipeline.predict(X_test_model)
    return predictions


def calculate_evaluation_metrics(
    pipelines: Dict[str, Pipeline],
    predictions: Dict[str, np.ndarray],
    X_test_model: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Calculate Accuracy, Precision, Recall, F1-Score, and ROC-AUC (Section 11.3).
    """
    evaluation_results = []
    y_test_binary = (y_test == "yes").astype(int)

    for name, pipeline in pipelines.items():
        y_pred = predictions[name]

        classes = list(pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index("yes")
        y_probability = pipeline.predict_proba(X_test_model)[:, positive_index]

        evaluation_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, pos_label="yes", zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, pos_label="yes", zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, y_pred, pos_label="yes", zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                y_test_binary, y_probability
            )
        })

    return pd.DataFrame(evaluation_results)


def plot_confusion_matrices(
    predictions: Dict[str, np.ndarray],
    y_test: pd.Series,
    show_plot: bool = True
) -> None:
    """
    Plot confusion matrices for all four models side by side (Section 11.4).
    """
    if not show_plot:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=["no", "yes"])
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        )
        display.plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(name)

    plt.tight_layout()
    plt.show()


def print_classification_reports(
    predictions: Dict[str, np.ndarray],
    y_test: pd.Series
) -> None:
    """
    Print detailed classification reports (Section 11.5).
    """
    for name, y_pred in predictions.items():
        print(f"\n{'=' * 60}")
        print(f"Classification Report: {name}")
        print("=" * 60)
        print(
            classification_report(
                y_test,
                y_pred,
                labels=["no", "yes"],
                target_names=["No", "Yes"],
                zero_division=0
            )
        )


def plot_roc_curves(
    pipelines: Dict[str, Pipeline],
    X_test_model: pd.DataFrame,
    y_test: pd.Series,
    show_plot: bool = True
) -> None:
    """
    Plot combined ROC curves (Section 11.6).
    """
    if not show_plot:
        return

    plt.figure(figsize=(9, 7))
    y_test_binary = (y_test == "yes").astype(int)

    for name, pipeline in pipelines.items():
        classes = list(pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index("yes")
        y_probability = pipeline.predict_proba(X_test_model)[:, positive_index]

        fpr, tpr, _ = roc_curve(y_test_binary, y_probability)
        auc_val = roc_auc_score(y_test_binary, y_probability)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    print("=" * 70)
    print("Section 11: Model Evaluation")
    print("=" * 70)

    # Prepare data and train baseline models
    df_raw = load_raw_dataset()
    df = apply_feature_engineering(df_raw)
    X, y = get_feature_target_split(df, TARGET)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_model, X_test_model = prepare_model_inputs(X_train, X_test)

    print("\nTraining baseline models for evaluation...")
    pipelines = build_model_pipelines()
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train_model, y_train)

    # 11.2 Generate Predictions
    predictions = generate_predictions(pipelines, X_test_model)
    print("\n11.2 Predictions generated for all 4 models.")

    # 11.3 Calculate Metrics
    evaluation_df = calculate_evaluation_metrics(
        pipelines, predictions, X_test_model, y_test
    )
    print("\n11.3 Evaluation Metrics Summary:")
    print(evaluation_df.round(4).to_string(index=False))

    # 11.4 Confusion Matrices
    print("\n11.4 Displaying Confusion Matrices...")
    plot_confusion_matrices(predictions, y_test, show_plot=True)

    # 11.5 Classification Reports
    print_classification_reports(predictions, y_test)

    # 11.6 ROC Curves
    print("\n11.6 Displaying ROC Curves...")
    plot_roc_curves(pipelines, X_test_model, y_test, show_plot=True)

    # 11.7 Sorted by F1-Score
    print("\n11.7 Models Ranked by F1-Score:")
    ranked_df = evaluation_df.sort_values(
        by="F1-Score", ascending=False
    ).reset_index(drop=True)
    print(ranked_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""
Section 16: Conclusion and Future Work
Corresponds to Cells 203-216 of `notebooks/bank_marketing_ml.ipynb`.

Synthesizes the complete Bank Marketing Machine Learning project:
- 16.1 Project metadata and scope summary.
- 16.2 Final model performance metrics (F1 = 0.4797, ROC-AUC = 0.8010).
- 16.3 Baseline vs. Tuned Random Forest comparison.
- 16.4 Practical interpretation of the confusion matrix.
- 16.5 Summary of influential features (balance, month, age, day, contact, poutcome).
- 16.6 Real-world project limitations.
- 16.7 Recommended future improvements and next steps.
- 16.8 Final project conclusion.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_project_summary_table(selected_model_name: str = "Random Forest") -> pd.DataFrame:
    """
    Construct Section 16.1 project overview table.
    """
    return pd.DataFrame({
        "Project Detail": [
            "Problem Type",
            "Dataset",
            "Original Records",
            "Original Columns",
            "Baseline Models",
            "Tuned Candidate Models",
            "Selected Model"
        ],
        "Value": [
            "Binary Classification",
            "bank-full.csv",
            45211,
            17,
            4,
            3,
            selected_model_name
        ]
    })


def build_confusion_summary_table(tn: int, fp: int, fn: int, tp: int) -> pd.DataFrame:
    """
    Construct Section 16.4 confusion matrix interpretation table.
    """
    return pd.DataFrame({
        "Outcome": [
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives"
        ],
        "Count": [tn, fp, fn, tp],
        "Meaning": [
            "Non-subscribers correctly predicted as no",
            "Non-subscribers incorrectly predicted as yes",
            "Subscribers incorrectly predicted as no",
            "Subscribers correctly predicted as yes"
        ]
    })


def print_project_limitations() -> None:
    """
    Print the six documented limitations from Section 16.6.
    """
    print("\n16.6 Project Limitations:")
    print("  1. The target is imbalanced (~88% non-subscribers), so accuracy alone is insufficient.")
    print("  2. The tuned model trades precision for higher recall, resulting in more false positives.")
    print("  3. The test set was inspected during earlier model comparison and is not a completely untouched benchmark.")
    print("  4. Feature importance reflects predictive correlation, not causal relationships.")
    print("  5. The optimal decision threshold has not yet been aligned with bank-specific financial contact costs.")
    print("  6. The model has not been validated through an actual A/B operational marketing campaign.")


def print_future_work() -> None:
    """
    Print recommended next steps from Section 16.7.
    """
    print("\n16.7 Future Work & Next Steps:")
    print("  - Evaluate the pipeline on a fresh, completely untouched holdout dataset.")
    print("  - Select an optimal probability decision threshold based on business contact costs vs. subscription value.")
    print("  - Investigate probability calibration if confidence scores will guide advisor workflows.")
    print("  - Evaluate fairness and consistency across customer demographic groups.")
    print("  - Build a clean, interactive Streamlit predictor UI for non-technical stakeholders.")
    print("  - Monitor prediction drift and model degradation after deployment.")


def main() -> None:
    print("=" * 70)
    print("Section 16: Project Conclusion & Future Work")
    print("=" * 70)

    # 16.1 Project Summary
    summary_df = build_project_summary_table()
    print("\n16.1 Project Overview:")
    print(summary_df.to_string(index=False))

    # 16.2 Final Metrics from Notebook Execution
    final_metrics_df = pd.DataFrame([{
        "Model": "Random Forest (Tuned)",
        "Accuracy": 0.8540,
        "Precision": 0.4287,
        "Recall": 0.5444,
        "F1-Score": 0.4797,
        "ROC-AUC": 0.8010
    }])
    print("\n16.2 Final Model Performance (Held-out Test Dataset):")
    print(final_metrics_df.round(4).to_string(index=False))

    # 16.3 Baseline vs. Tuned Comparison
    comparison_df = pd.DataFrame([
        {"Model": "Baseline Random Forest", "Accuracy": 0.8931, "Precision": 0.5976, "Recall": 0.2325, "F1-Score": 0.3349, "ROC-AUC": 0.7960},
        {"Model": "Tuned Random Forest", "Accuracy": 0.8540, "Precision": 0.4287, "Recall": 0.5444, "F1-Score": 0.4797, "ROC-AUC": 0.8010}
    ])
    print("\n16.3 Baseline vs. Tuned Model Comparison:")
    print(comparison_df.round(4).to_string(index=False))
    print("  -> The tuned model substantially improved recall (23.3% -> 54.4%) and F1-score (0.335 -> 0.480).")

    # 16.4 Confusion Matrix Interpretation
    # Test set size: 9,043 samples
    confusion_table = build_confusion_summary_table(
        tn=7147, fp=845, fn=480, tp=571
    )
    print("\n16.4 Confusion Matrix Practical Breakdown:")
    print(confusion_table.to_string(index=False))

    # 16.5 Key Influential Features
    print("\n16.5 Influential Features Summary:")
    print("  - Customer balance, contact month, customer age, day of month, contact method, and previous campaign outcome.")

    # 16.6 Limitations
    print_project_limitations()

    # 16.7 Future Work
    print_future_work()

    # 16.8 Final Conclusion
    print("\n16.8 Final Conclusion:")
    print("  This project developed an end-to-end machine learning pipeline predicting bank term deposit")
    print("  subscriptions without relying on the post-contact 'duration' feature. The tuned Random Forest")
    print("  achieved strong ranking ability (ROC-AUC 0.801) and balanced detection (F1 0.480), demonstrating")
    print("  clear feasibility for pre-campaign marketing prioritization.")


if __name__ == "__main__":
    main()

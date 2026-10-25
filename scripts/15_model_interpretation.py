"""
Section 15: Model Interpretation
Corresponds to Cells 185-202 of `notebooks/bank_marketing_ml.ipynb`.

Analyzes predictive feature contributions for the selected Random Forest model using:
1. Built-in Gini / impurity-based feature importances (raw transformed and aggregated).
2. Permutation feature importances based on decrease in test F1-Score.
3. Comparative ranking analysis between both interpretation methods.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    MODEL_CATEGORICAL_FEATURES,
    TARGET,
    apply_feature_engineering,
    create_preprocessor,
    get_feature_target_split,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)
from sklearn.ensemble import RandomForestClassifier


def get_original_feature(transformed_name: str, categorical_features: List[str]) -> str:
    """
    Map an encoded feature name (e.g., 'cat__job_admin.') back to its original column name ('job').
    Matches Section 15.4 logic.
    """
    if transformed_name.startswith("num__"):
        return transformed_name[len("num__"):]

    if transformed_name.startswith("cat__"):
        encoded_name = transformed_name[len("cat__"):]
        matches = [
            feature
            for feature in categorical_features
            if encoded_name.startswith(feature + "_") or encoded_name == feature
        ]
        if matches:
            return max(matches, key=len)

    return transformed_name


def extract_built_in_importance(
    final_model: Pipeline,
    categorical_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract raw transformed feature importances and aggregated original feature importances (Sections 15.2 & 15.4).
    """
    preprocessor = final_model.named_steps["preprocessor"]
    classifier = final_model.named_steps["classifier"]

    transformed_feature_names = preprocessor.get_feature_names_out()
    feature_importances = classifier.feature_importances_

    assert len(transformed_feature_names) == len(feature_importances)

    importance_df = pd.DataFrame({
        "Feature": transformed_feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    importance_df["Original Feature"] = importance_df["Feature"].apply(
        lambda name: get_original_feature(name, categorical_features)
    )

    aggregated_importance = (
        importance_df.groupby("Original Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df, aggregated_importance


def compute_permutation_importance(
    final_model: Pipeline,
    X_test_model: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance using test F1-score drop (Section 15.6).
    """
    f1_scorer = make_scorer(f1_score, pos_label="yes", zero_division=0)

    result = permutation_importance(
        estimator=final_model,
        X=X_test_model,
        y=y_test,
        scoring=f1_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1
    )

    perm_df = pd.DataFrame({
        "Feature": X_test_model.columns,
        "Importance Mean": result.importances_mean,
        "Importance Std": result.importances_std
    }).sort_values(by="Importance Mean", ascending=False).reset_index(drop=True)

    return perm_df


def compare_importance_rankings(
    aggregated_importance: pd.DataFrame,
    permutation_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge and compare built-in and permutation importance rankings (Section 15.8).
    """
    built_in_ranking = aggregated_importance.rename(
        columns={"Original Feature": "Feature", "Importance": "Built-in Importance"}
    )
    merged = built_in_ranking.merge(
        permutation_df[["Feature", "Importance Mean", "Importance Std"]],
        on="Feature",
        how="outer"
    ).sort_values(by="Importance Mean", ascending=False).reset_index(drop=True)
    return merged


def plot_feature_importances(
    importance_df: pd.DataFrame,
    aggregated_importance: pd.DataFrame,
    permutation_df: pd.DataFrame,
    show_plot: bool = True
) -> None:
    """
    Generate plots for Section 15.3, 15.5, and 15.7.
    """
    if not show_plot:
        return

    # 15.3 Top 15 Transformed
    top_trans = importance_df.head(15).sort_values(by="Importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_trans["Feature"], top_trans["Importance"])
    plt.title("Top 15 Transformed Feature Importances (Built-in)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    # 15.5 Top Aggregated Original
    top_orig = aggregated_importance.head(15).sort_values(by="Importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_orig["Original Feature"], top_orig["Importance"])
    plt.title("Importance by Original Feature (Aggregated Built-in)")
    plt.xlabel("Aggregated Importance")
    plt.tight_layout()
    plt.show()

    # 15.7 Permutation Importance
    top_perm = permutation_df.head(15).sort_values(by="Importance Mean", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_perm["Feature"],
        top_perm["Importance Mean"],
        xerr=top_perm["Importance Std"],
        capsize=3
    )
    plt.axvline(0, linestyle="--", color="black", alpha=0.5)
    plt.title("Permutation Importance — Mean F1 Drop on Test Set")
    plt.xlabel("Mean Decrease in F1-Score")
    plt.tight_layout()
    plt.show()


def main() -> None:
    print("=" * 70)
    print("Section 15: Model Interpretation")
    print("=" * 70)

    # Prepare data and fit model
    df_raw = load_raw_dataset()
    df = apply_feature_engineering(df_raw)
    X, y = get_feature_target_split(df, TARGET)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_model, X_test_model = prepare_model_inputs(X_train, X_test)

    print("\nFitting Random Forest pipeline for interpretation...")
    final_model = Pipeline([
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
    final_model.fit(X_train_model, y_train)

    # 15.2 & 15.4 Built-in Importance
    importance_df, aggregated_df = extract_built_in_importance(
        final_model, MODEL_CATEGORICAL_FEATURES
    )
    print("\n15.4 Top 10 Original Features by Built-in Importance:")
    print(aggregated_df.head(10).round(4).to_string(index=False))

    # 15.6 Permutation Importance
    print("\n15.6 Computing Permutation Importance on Test Set...")
    perm_df = compute_permutation_importance(final_model, X_test_model, y_test)
    print("\nTop 10 Features by Permutation Importance (F1 drop):")
    print(perm_df.head(10).round(4).to_string(index=False))

    # 15.8 Compare Two Importance Methods
    print("\n15.8 Comparison of Feature Importance Methods:")
    comparison_df = compare_importance_rankings(aggregated_df, perm_df)
    print(comparison_df.round(4).to_string(index=False))

    # Plots
    print("\nDisplaying Interpretation Plots...")
    plot_feature_importances(importance_df, aggregated_df, perm_df, show_plot=True)


if __name__ == "__main__":
    main()

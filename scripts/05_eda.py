"""
Section 6: Exploratory Data Analysis (EDA)
Corresponds to Cells 53-71 of `notebooks/bank_marketing_ml.ipynb`.

Generates visualizations and summary statistics for:
- 6.1 Target distribution and class imbalance
- 6.2 Numerical feature distributions (age, balance, duration, campaign)
- 6.3 Categorical feature distributions (job, education, contact, poutcome)
- 6.4 Subscription rate across categorical features
- 6.5 Numerical features by subscription outcome (age, balance boxplots)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import TARGET, load_raw_dataset


def plot_target_distribution(df: pd.DataFrame, show_plot: bool = True) -> pd.Series:
    """
    Plot and calculate target distribution percentages (Section 6.1).
    """
    target_percentages = (
        df[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    if show_plot:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=TARGET)
        plt.title("Distribution of Subscription Outcome")
        plt.xlabel("Subscription Outcome")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()

    return target_percentages


def plot_numerical_distributions(df: pd.DataFrame, show_plot: bool = True) -> None:
    """
    Plot histograms for selected numerical features (Section 6.2).
    """
    if not show_plot:
        return

    selected_numerical = ["age", "balance", "duration", "campaign"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, column in zip(axes.flat, selected_numerical):
        sns.histplot(data=df, x=column, bins=30, ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df: pd.DataFrame, show_plot: bool = True) -> None:
    """
    Plot countplots for selected categorical features (Section 6.3).
    """
    if not show_plot:
        return

    selected_categorical = ["job", "education", "contact", "poutcome"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, column in zip(axes.flat, selected_categorical):
        order = df[column].value_counts().index
        sns.countplot(data=df, x=column, order=order, ax=ax)
        ax.set_title(f"Customer Count by {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Number of Customers")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_subscription_rates(df: pd.DataFrame, show_plot: bool = True) -> dict:
    """
    Calculate and plot subscription rate (%) across categorical features (Section 6.4).
    """
    selected_rate_features = ["job", "marital", "education", "contact", "poutcome"]
    df_eda = df.assign(subscribed=(df[TARGET] == "yes").astype(int))
    rate_tables = {}

    for column in selected_rate_features:
        rates = (
            df_eda.groupby(column)["subscribed"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        rate_tables[column] = rates

    if show_plot:
        fig, axes = plt.subplots(3, 2, figsize=(14, 13))

        for ax, column in zip(axes.flat, selected_rate_features):
            rates = rate_tables[column]
            sns.barplot(x=rates.values, y=rates.index, ax=ax)
            ax.set_title(f"Subscription Rate by {column}")
            ax.set_xlabel("Subscription Rate (%)")
            ax.set_ylabel(column)

        fig.delaxes(axes.flat[-1])
        plt.tight_layout()
        plt.show()

    return rate_tables


def plot_numerical_by_outcome(df: pd.DataFrame, show_plot: bool = True) -> None:
    """
    Plot boxplots of age and balance by subscription outcome (Section 6.5).
    """
    if not show_plot:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=df, x=TARGET, y="age", ax=axes[0])
    axes[0].set_title("Age by Subscription Outcome")
    axes[0].set_xlabel("Subscription Outcome")
    axes[0].set_ylabel("Age")

    sns.boxplot(data=df, x=TARGET, y="balance", ax=axes[1])
    axes[1].set_title("Balance by Subscription Outcome")
    axes[1].set_xlabel("Subscription Outcome")
    axes[1].set_ylabel("Balance")

    plt.tight_layout()
    plt.show()


def run_eda(df: pd.DataFrame, show_plots: bool = True) -> None:
    """
    Run complete exploratory analysis matching Section 6.
    """
    sns.set_theme(style="whitegrid")

    print("\n--- 6.1 Target Distribution ---")
    percentages = plot_target_distribution(df, show_plot=show_plots)
    print(percentages)

    print("\n--- 6.2 Numerical Feature Distributions ---")
    plot_numerical_distributions(df, show_plot=show_plots)
    print("Displayed distributions for age, balance, duration, campaign.")

    print("\n--- 6.3 Categorical Feature Distributions ---")
    plot_categorical_distributions(df, show_plot=show_plots)
    print("Displayed counts for job, education, contact, poutcome.")

    print("\n--- 6.4 Subscription Rates by Category ---")
    rates = plot_subscription_rates(df, show_plot=show_plots)
    for col, r in rates.items():
        print(f"Top subscription rates for {col}:\n{r.head(3).round(2)}\n")

    print("\n--- 6.5 Numerical Features by Subscription Outcome ---")
    plot_numerical_by_outcome(df, show_plot=show_plots)
    print("Displayed boxplots for age and balance across subscription outcomes.")

    print("\n--- 6.6 EDA Summary ---")
    print("  - Target is imbalanced (88.3% 'no', 11.7% 'yes').")
    print("  - Numerical features (balance, duration, campaign) are highly right-skewed.")
    print("  - Notable subscription variation across categories (e.g. poutcome='success' has highest conversion).")


def main() -> None:
    print("=" * 70)
    print("Section 6: Exploratory Data Analysis (EDA)")
    print("=" * 70)

    df = load_raw_dataset()
    # By default, when executed directly, run EDA
    run_eda(df, show_plots=True)


if __name__ == "__main__":
    main()

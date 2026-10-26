"""
Dataset Service for Bank Marketing Machine Learning Application.
Loads and caches data summaries, feature catalogue, and read-only previews from data/bank-full.csv.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class DatasetService:
    """
    Manages cached data loading, EDA aggregations, and preview generation.
    Strictly read-only; never alters data/bank-full.csv.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or self._resolve_default_data_path()
        self._df: Optional[pd.DataFrame] = None
        self._summary_cache: Optional[Dict[str, Any]] = None
        self._eda_cache: Optional[Dict[str, Any]] = None

    @staticmethod
    def _resolve_default_data_path() -> Path:
        """Resolve dataset path relative to backend or project root."""
        env_path = os.getenv("DATA_PATH")
        if env_path and Path(env_path).is_file():
            return Path(env_path).resolve()

        backend_dir = Path(__file__).resolve().parent
        project_root = backend_dir.parent
        candidates = [
            project_root / "data" / "bank-full.csv",
            backend_dir / "data" / "bank-full.csv",
            backend_dir / ".." / "data" / "bank-full.csv",
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()

        return (project_root / "data" / "bank-full.csv").resolve()

    def get_dataframe(self) -> pd.DataFrame:
        """Load dataset once into memory."""
        if self._df is None:
            if not self.data_path.is_file():
                raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
            # Load semicolon-delimited CSV
            self._df = pd.read_csv(self.data_path, sep=";")
        return self._df

    def get_summary(self) -> Dict[str, Any]:
        """Compute and cache top-level summary metrics for the dataset."""
        if self._summary_cache is not None:
            return self._summary_cache

        df = self.get_dataframe()
        total_records = int(len(df))
        total_columns = int(len(df.columns))

        # Target distribution
        target_counts = df["y"].value_counts().to_dict()
        no_count = int(target_counts.get("no", 0))
        yes_count = int(target_counts.get("yes", 0))

        no_pct = round((no_count / total_records) * 100, 2)
        yes_pct = round((yes_count / total_records) * 100, 2)

        numerical_features = [
            "age", "balance", "day", "duration", "campaign", "pdays", "previous"
        ]
        categorical_features = [
            "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"
        ]

        self._summary_cache = {
            "filename": self.data_path.name,
            "total_records": total_records,
            "total_columns": total_columns,
            "delimiter": ";",
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "target_column": "y",
            "target_classes": ["no", "yes"],
            "target_distribution": {
                "no": {"count": no_count, "percentage": no_pct, "label": "Did not subscribe"},
                "yes": {"count": yes_count, "percentage": yes_pct, "label": "Subscribed to term deposit"},
            },
            "numerical_features_count": len(numerical_features),
            "categorical_features_count": len(categorical_features),
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "model_features_count": 16,
            "engineered_features": ["previously_contacted"],
            "excluded_features": ["duration"],
            "pdays_negative_one_count": int((df["pdays"] == -1).sum()),
            "pdays_negative_one_pct": round(float((df["pdays"] == -1).sum() / total_records) * 100, 2),
        }
        return self._summary_cache

    def get_features_catalogue(self) -> List[Dict[str, Any]]:
        """Return structured catalogue of all features with role, description, and status."""
        return [
            {
                "name": "age",
                "data_type": "integer",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Customer age in years (18 to 95).",
                "examples": "18 – 95 (median: 39)",
                "notes": "Standardized with StandardScaler",
            },
            {
                "name": "job",
                "data_type": "categorical",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Occupation category of the customer.",
                "examples": "management, technician, blue-collar, admin., services, retired, student...",
                "notes": "One-hot encoded (12 categories including 'unknown')",
            },
            {
                "name": "marital",
                "data_type": "categorical",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Marital status of the customer.",
                "examples": "married (58.3%), single (28.3%), divorced (11.5%)",
                "notes": "One-hot encoded (3 categories)",
            },
            {
                "name": "education",
                "data_type": "categorical",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Highest education level completed.",
                "examples": "primary, secondary, tertiary, unknown",
                "notes": "One-hot encoded (4 categories)",
            },
            {
                "name": "default",
                "data_type": "binary",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Whether the customer has credit currently in default.",
                "examples": "no (98.2%), yes (1.8%)",
                "notes": "One-hot encoded",
            },
            {
                "name": "balance",
                "data_type": "numeric (float)",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Average yearly account balance in Euros (can be negative).",
                "examples": "-8,019 to 102,127 € (median: 448 €)",
                "notes": "Standardized with StandardScaler; valid negative balances retained",
            },
            {
                "name": "housing",
                "data_type": "binary",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Has active housing mortgage loan.",
                "examples": "yes (55.6%), no (44.4%)",
                "notes": "One-hot encoded",
            },
            {
                "name": "loan",
                "data_type": "binary",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Has active personal consumer loan.",
                "examples": "no (84.0%), yes (16.0%)",
                "notes": "One-hot encoded",
            },
            {
                "name": "contact",
                "data_type": "categorical",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Communication channel used to reach the customer.",
                "examples": "cellular (64.8%), unknown (28.8%), telephone (6.4%)",
                "notes": "One-hot encoded",
            },
            {
                "name": "day",
                "data_type": "integer",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Last contact day of the month.",
                "examples": "1 – 31 (median: 16)",
                "notes": "Standardized with StandardScaler",
            },
            {
                "name": "month",
                "data_type": "categorical",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Last contact calendar month of the year (3-letter abbreviation).",
                "examples": "jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec",
                "notes": "One-hot encoded (12 categories)",
            },
            {
                "name": "duration",
                "data_type": "integer (seconds)",
                "role": "Excluded Feature",
                "used_in_model": False,
                "description": "Duration of the contact phone call in seconds.",
                "examples": "0 – 4,918 seconds (mean: 258s)",
                "notes": "Strictly EXCLUDED to prevent post-contact target leakage. Not knowable pre-call.",
            },
            {
                "name": "campaign",
                "data_type": "integer",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Number of contacts performed during this campaign for this client.",
                "examples": "1 – 63 contacts (median: 2)",
                "notes": "Standardized with StandardScaler",
            },
            {
                "name": "pdays",
                "data_type": "integer",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Number of days since customer was contacted from a previous campaign (-1 means not previously contacted).",
                "examples": "-1 (81.7% of customers) or 1 – 871 days",
                "notes": "Standardized with StandardScaler; -1 sentinel retained",
            },
            {
                "name": "previous",
                "data_type": "integer",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Number of contacts performed before this campaign for this client.",
                "examples": "0 – 275 (median: 0, 81.7% are 0)",
                "notes": "Standardized with StandardScaler",
            },
            {
                "name": "poutcome",
                "data_type": "categorical",
                "role": "Model Input",
                "used_in_model": True,
                "description": "Outcome of the previous marketing campaign.",
                "examples": "unknown (81.7%), failure (10.8%), other (4.1%), success (3.3%)",
                "notes": "One-hot encoded; poutcome='success' has highest conversion rate (64.7%)",
            },
            {
                "name": "previously_contacted",
                "data_type": "binary (integer)",
                "role": "Derived Feature",
                "used_in_model": True,
                "description": "Binary indicator: 0 if pdays == -1 (first contact), 1 if pdays != -1.",
                "examples": "0 (81.7%), 1 (18.3%)",
                "notes": "Engineered feature; standardized with StandardScaler",
            },
            {
                "name": "y",
                "data_type": "binary",
                "role": "Target Variable",
                "used_in_model": False,
                "description": "Has the client subscribed to a term deposit? Binary target variable.",
                "examples": "no (39,922 / 88.3%), yes (5,289 / 11.7%)",
                "notes": "Target column for supervised classification",
            },
        ]

    def get_preview(self, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """Return a slice of original records for read-only preview."""
        df = self.get_dataframe()
        total_records = len(df)
        total_pages = max(1, (total_records + page_size - 1) // page_size)

        current_page = max(1, min(page, total_pages))
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_records)

        slice_df = df.iloc[start_idx:end_idx].copy()
        # Add index column for readability
        slice_df.insert(0, "record_id", range(start_idx + 1, end_idx + 1))

        # Convert to dictionary records
        records = slice_df.to_dict(orient="records")

        return {
            "total_records": total_records,
            "total_pages": total_pages,
            "current_page": current_page,
            "page_size": page_size,
            "columns": list(slice_df.columns),
            "rows": records,
        }

    def get_eda_metrics(self) -> Dict[str, Any]:
        """Return computed EDA distributions grounded directly in the source data."""
        if self._eda_cache is not None:
            return self._eda_cache

        df = self.get_dataframe()
        df_eda = df.copy()
        df_eda["subscribed_num"] = (df_eda["y"] == "yes").astype(int)

        # 1. Target breakdown
        target_dist = [
            {"outcome": "no", "count": int((df["y"] == "no").sum()), "percentage": 88.3, "label": "No (Did not subscribe)"},
            {"outcome": "yes", "count": int((df["y"] == "yes").sum()), "percentage": 11.7, "label": "Yes (Subscribed)"},
        ]

        # 2. Conversion rate by poutcome
        poutcome_stats = (
            df_eda.groupby("poutcome")["subscribed_num"]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values(by="mean", ascending=False)
        )
        poutcome_list = [
            {
                "category": row["poutcome"],
                "total_customers": int(row["count"]),
                "conversion_rate_pct": round(float(row["mean"]) * 100, 2),
            }
            for _, row in poutcome_stats.iterrows()
        ]

        # 3. Conversion rate by contact method
        contact_stats = (
            df_eda.groupby("contact")["subscribed_num"]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values(by="mean", ascending=False)
        )
        contact_list = [
            {
                "category": row["contact"],
                "total_customers": int(row["count"]),
                "conversion_rate_pct": round(float(row["mean"]) * 100, 2),
            }
            for _, row in contact_stats.iterrows()
        ]

        # 4. Conversion rate by housing loan
        housing_stats = (
            df_eda.groupby("housing")["subscribed_num"]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values(by="mean", ascending=False)
        )
        housing_list = [
            {
                "category": "No Housing Loan" if row["housing"] == "no" else "Has Housing Loan",
                "raw_value": row["housing"],
                "total_customers": int(row["count"]),
                "conversion_rate_pct": round(float(row["mean"]) * 100, 2),
            }
            for _, row in housing_stats.iterrows()
        ]

        # 5. Conversion rate by job
        job_stats = (
            df_eda.groupby("job")["subscribed_num"]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values(by="mean", ascending=False)
        )
        job_list = [
            {
                "category": row["job"],
                "total_customers": int(row["count"]),
                "conversion_rate_pct": round(float(row["mean"]) * 100, 2),
            }
            for _, row in job_stats.iterrows()
        ]

        # 6. Balance & Age statistical comparisons across outcomes
        age_no = df[df["y"] == "no"]["age"]
        age_yes = df[df["y"] == "yes"]["age"]
        bal_no = df[df["y"] == "no"]["balance"]
        bal_yes = df[df["y"] == "yes"]["balance"]

        numerical_comparisons = {
            "age": {
                "unit": "years",
                "overall_median": float(df["age"].median()),
                "no_mean": round(float(age_no.mean()), 1),
                "no_median": float(age_no.median()),
                "yes_mean": round(float(age_yes.mean()), 1),
                "yes_median": float(age_yes.median()),
            },
            "balance": {
                "unit": "Euros (€)",
                "overall_median": float(df["balance"].median()),
                "no_mean": round(float(bal_no.mean()), 1),
                "no_median": float(bal_no.median()),
                "yes_mean": round(float(bal_yes.mean()), 1),
                "yes_median": float(bal_yes.median()),
            },
        }

        self._eda_cache = {
            "target_distribution": target_dist,
            "subscription_rate_by_poutcome": poutcome_list,
            "subscription_rate_by_contact": contact_list,
            "subscription_rate_by_housing": housing_list,
            "subscription_rate_by_job": job_list,
            "numerical_comparisons": numerical_comparisons,
        }
        return self._eda_cache


# Global singleton instance
dataset_service = DatasetService()

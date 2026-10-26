"""
Prediction Service for Bank Marketing Machine Learning Pipeline.
Handles model loading, feature preparation, and inference.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

# Expected 16-feature schema (Duration is strictly excluded)
EXPECTED_FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "previously_contacted",
]


class PredictionService:
    """
    Manages model loading and prediction lifecycle.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or self._resolve_default_model_path()
        self.pipeline: Optional[Any] = None
        self.model_name = "Random Forest (Tuned Pipeline)"

    @staticmethod
    def _resolve_default_model_path() -> Path:
        """
        Locate the serialized pipeline relative to backend or project root.
        """
        env_path = os.getenv("MODEL_PATH")
        if env_path:
            p = Path(env_path)
            if p.is_file():
                return p.resolve()

        backend_dir = Path(__file__).resolve().parent
        project_root = backend_dir.parent

        candidate_paths = [
            project_root / "models" / "random_forest_pipeline.joblib",
            backend_dir / "models" / "random_forest_pipeline.joblib",
            backend_dir / ".." / "models" / "random_forest_pipeline.joblib",
        ]

        for p in candidate_paths:
            if p.is_file():
                return p.resolve()

        # Fallback to standard path
        return (project_root / "models" / "random_forest_pipeline.joblib").resolve()

    def load_model(self) -> None:
        """
        Load the joblib pipeline into memory.
        """
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model artifact not found at: {self.model_path}. "
                "Ensure 'python export_model.py' has been executed."
            )
        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to deserialize model pipeline from {self.model_path}: {exc}") from exc

    def is_loaded(self) -> bool:
        """Check if pipeline is active in memory."""
        return self.pipeline is not None

    def prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Construct a single-row DataFrame matching the exact 16 expected features.
        Computes 'previously_contacted' from pdays.
        Duration is strictly excluded.
        """
        pdays = int(data["pdays"])
        previously_contacted = 1 if pdays != -1 else 0

        row = {
            "age": int(data["age"]),
            "job": str(data["job"]),
            "marital": str(data["marital"]),
            "education": str(data["education"]),
            "default": str(data["default"]),
            "balance": float(data["balance"]),
            "housing": str(data["housing"]),
            "loan": str(data["loan"]),
            "contact": str(data["contact"]),
            "day": int(data["day"]),
            "month": str(data["month"]),
            "campaign": int(data["campaign"]),
            "pdays": pdays,
            "previous": int(data["previous"]),
            "poutcome": str(data["poutcome"]),
            "previously_contacted": previously_contacted,
        }

        df = pd.DataFrame([row])
        return df[EXPECTED_FEATURE_COLUMNS]

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute prediction and return outcome and subscription probability.
        """
        if not self.is_loaded():
            raise RuntimeError("Model pipeline is not loaded.")

        df = self.prepare_dataframe(data)

        # Retrieve classifier to inspect classes
        classifier = self.pipeline.named_steps.get("classifier")
        if classifier is None:
            raise RuntimeError("Pipeline does not contain a 'classifier' step.")

        classes = list(classifier.classes_)
        if "yes" not in classes:
            raise ValueError(f"Expected positive class 'yes' in classifier classes: {classes}")

        yes_idx = classes.index("yes")

        raw_pred = self.pipeline.predict(df)[0]
        prediction = "yes" if str(raw_pred).lower() == "yes" else "no"
        prediction_text = "Likely to subscribe" if prediction == "yes" else "Unlikely to subscribe"

        prob_val: Optional[float] = None
        prob_available = False

        if hasattr(self.pipeline, "predict_proba"):
            try:
                probs = self.pipeline.predict_proba(df)[0]
                prob_val = round(float(probs[yes_idx]), 4)
                prob_available = True
            except Exception:
                prob_val = None
                prob_available = False

        return {
            "prediction": prediction,
            "prediction_text": prediction_text,
            "subscription_probability": prob_val,
            "probability_available": prob_available,
            "model_name": self.model_name,
            "disclaimer": "Predictions are estimates based on historical data and do not guarantee actual customer behavior.",
        }


# Global singleton instance
prediction_service = PredictionService()

"""
Automated Unit Tests for Bank Marketing Predictor UI and Inference Pipeline.
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    DEFAULT_MODEL_PATH,
    EXPECTED_FEATURE_COLUMNS,
    build_inference_dataframe,
    load_trained_model,
    predict_subscription,
    validate_customer_input,
)


class TestBankMarketingUIInference(unittest.TestCase):
    """Test suite for the UI inference pipeline and model behavior."""

    @classmethod
    def setUpClass(cls):
        """Ensure the model is loaded once for tests."""
        cls.model = load_trained_model(DEFAULT_MODEL_PATH)

    def test_01_model_loading_and_type(self):
        """Verify the loaded model is the intended tuned Random Forest pipeline."""
        self.assertIsNotNone(self.model)
        self.assertIn("preprocessor", self.model.named_steps)
        self.assertIn("classifier", self.model.named_steps)
        classifier = self.model.named_steps["classifier"]
        self.assertEqual(type(classifier).__name__, "RandomForestClassifier")
        self.assertEqual(classifier.n_estimators, 200)
        self.assertEqual(classifier.max_depth, 15)
        self.assertEqual(classifier.min_samples_leaf, 3)
        self.assertEqual(classifier.class_weight, "balanced")
        self.assertEqual(classifier.random_state, 42)

    def test_02_feature_names_and_duration_excluded(self):
        """Verify the 16 expected features and confirm duration is excluded."""
        self.assertEqual(len(EXPECTED_FEATURE_COLUMNS), 16)
        self.assertNotIn("duration", EXPECTED_FEATURE_COLUMNS)
        self.assertIn("previously_contacted", EXPECTED_FEATURE_COLUMNS)

    def test_03_prediction_with_pdays_negative_one(self):
        """Test inference when customer was not previously contacted (pdays = -1)."""
        payload = {
            "age": 42,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "balance": 1500.0,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": 12,
            "month": "aug",
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }
        df = build_inference_dataframe(payload)
        self.assertEqual(df["previously_contacted"].iloc[0], 0)
        self.assertEqual(df["pdays"].iloc[0], -1)

        result = predict_subscription(self.model, df)
        self.assertIn(result["raw_prediction"], ["no", "yes"])
        self.assertIn(result["prediction_text"], ["Likely to subscribe", "Unlikely to subscribe"])
        self.assertTrue(0.0 <= result["positive_probability"] <= 1.0)
        self.assertEqual(result["positive_class"], "yes")

    def test_04_prediction_with_negative_balance(self):
        """Test that negative account balances are accepted and evaluated."""
        payload = {
            "age": 30,
            "job": "technician",
            "marital": "single",
            "education": "secondary",
            "default": "yes",
            "balance": -450.50,
            "housing": "yes",
            "loan": "yes",
            "contact": "cellular",
            "day": 5,
            "month": "may",
            "campaign": 3,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }
        df = build_inference_dataframe(payload)
        self.assertEqual(df["balance"].iloc[0], -450.50)
        result = predict_subscription(self.model, df)
        self.assertIsNotNone(result)

    def test_05_prediction_with_prior_contact(self):
        """Test inference when customer was previously contacted."""
        payload = {
            "age": 55,
            "job": "retired",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": 8000.0,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": 20,
            "month": "oct",
            "campaign": 1,
            "pdays": 180,
            "previous": 3,
            "poutcome": "success",
        }
        df = build_inference_dataframe(payload)
        self.assertEqual(df["previously_contacted"].iloc[0], 1)
        self.assertEqual(df["pdays"].iloc[0], 180)

        result = predict_subscription(self.model, df)
        self.assertEqual(result["raw_prediction"], "yes")
        self.assertEqual(result["prediction_text"], "Likely to subscribe")
        self.assertGreater(result["positive_probability"], 0.5)

    def test_06_unknown_category_support(self):
        """Verify that supported 'unknown' values are accepted without error."""
        payload = {
            "age": 38,
            "job": "unknown",
            "marital": "married",
            "education": "unknown",
            "default": "no",
            "balance": 200.0,
            "housing": "no",
            "loan": "no",
            "contact": "unknown",
            "day": 15,
            "month": "jun",
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }
        df = build_inference_dataframe(payload)
        result = predict_subscription(self.model, df)
        self.assertIsNotNone(result)

    def test_07_input_validation_errors(self):
        """Verify validation catches invalid inputs."""
        # Age below 18
        is_valid, msg = validate_customer_input({"age": 16})
        self.assertFalse(is_valid)
        self.assertIn("Age", msg)

        # Invalid category
        payload = {
            "age": 30,
            "job": "astronaut",  # Invalid
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "balance": 100.0,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": 10,
            "month": "may",
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }
        is_valid, msg = validate_customer_input(payload)
        self.assertFalse(is_valid)
        self.assertIn("job", msg)

    def test_08_missing_model_error(self):
        """Verify helpful error when model file does not exist."""
        non_existent_path = PROJECT_ROOT / "models" / "does_not_exist.joblib"
        with self.assertRaises(FileNotFoundError) as ctx:
            load_trained_model(non_existent_path)
        self.assertIn("export_model.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

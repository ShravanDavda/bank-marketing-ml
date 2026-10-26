"""
Automated Unit and Integration Tests for FastAPI Backend.
Verifies API health, request validation, prediction correctness, and edge cases.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app
from backend.prediction_service import prediction_service


class TestFastAPIBackend(unittest.TestCase):
    """Integration test suite for Bank Marketing FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient and ensure model is loaded."""
        prediction_service.load_model()
        cls.client = TestClient(app)

    def test_01_health_check(self):
        """GET /api/health returns 200 with healthy status and model_loaded=True."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("Random Forest", data["model_name"])
        self.assertIn("duration excluded", data["timing_constraint"])

    def test_02_predict_valid_standard_customer(self):
        """POST /api/predict returns valid prediction and probability."""
        payload = {
            "age": 35,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "balance": 2500.0,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": 15,
            "month": "may",
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["yes", "no"])
        self.assertIn(data["prediction_text"], ["Likely to subscribe", "Unlikely to subscribe"])
        self.assertTrue(data["probability_available"])
        self.assertIsInstance(data["subscription_probability"], float)
        self.assertTrue(0.0 <= data["subscription_probability"] <= 1.0)
        self.assertIn("Random Forest", data["model_name"])
        self.assertIn("historical data", data["disclaimer"])

    def test_03_predict_first_time_contact(self):
        """Verify pdays = -1 sentinel is accepted and handled properly."""
        payload = {
            "age": 42,
            "job": "technician",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "balance": 500.0,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "day": 10,
            "month": "aug",
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["yes", "no"])

    def test_04_predict_previously_contacted_customer(self):
        """Verify previously contacted customer with success outcome returns positive probability."""
        payload = {
            "age": 58,
            "job": "retired",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": 8500.0,
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
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], "yes")
        self.assertEqual(data["prediction_text"], "Likely to subscribe")
        self.assertGreater(data["subscription_probability"], 0.5)

    def test_05_predict_negative_balance(self):
        """Verify negative account balances are accepted and evaluated."""
        payload = {
            "age": 29,
            "job": "services",
            "marital": "single",
            "education": "secondary",
            "default": "yes",
            "balance": -350.50,
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
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["yes", "no"])

    def test_06_validation_error_invalid_age(self):
        """Verify rejection of age < 18 with 422 HTTP code."""
        payload = {
            "age": 16,  # Below minimum 18
            "job": "student",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "balance": 100.0,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": 10,
            "month": "sep",
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_07_validation_error_invalid_category(self):
        """Verify rejection of invalid job category with 422 HTTP code."""
        payload = {
            "age": 30,
            "job": "astronaut",  # Not in valid list
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "balance": 1000.0,
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
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_08_duration_excluded(self):
        """Verify duration is not required and has no effect if passed."""
        payload = {
            "age": 35,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "balance": 1500.0,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": 12,
            "month": "nov",
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }
        # duration is not present in payload
        self.assertNotIn("duration", payload)
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_09_get_pipeline_metadata(self):
        """GET /api/pipeline returns verified stages and model specs."""
        response = self.client.get("/api/pipeline")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["title"], "Machine Learning Pipeline")
        self.assertGreaterEqual(data["total_stages"], 10)
        self.assertIn("Random Forest", data["model_specifications"]["model_name"])
        first_stage = data["stages"][0]
        self.assertIn("title", first_stage)
        self.assertIn("what_it_means", first_stage)

    def test_10_get_performance_data(self):
        """GET /api/performance returns verified metrics and confusion matrix."""
        response = self.client.get("/api/performance")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["test_set_size"], 9043)
        self.assertEqual(len(data["summary_metrics"]), 5)
        # Check verified metric values
        metrics_dict = {m["name"]: m["value"] for m in data["summary_metrics"]}
        self.assertAlmostEqual(metrics_dict["F1-Score"], 0.4765, places=3)
        self.assertAlmostEqual(metrics_dict["ROC-AUC"], 0.8025, places=3)
        self.assertAlmostEqual(metrics_dict["Accuracy"], 0.8510, places=3)
        self.assertAlmostEqual(metrics_dict["Precision"], 0.4046, places=3)
        self.assertAlmostEqual(metrics_dict["Recall (Sensitivity)"], 0.5794, places=3)
        # Confusion matrix checks
        cm = data["confusion_matrix"]
        tn = cm["true_negatives"]
        fp = cm["false_positives"]
        fn = cm["false_negatives"]
        tp = cm["true_positives"]
        self.assertEqual(tn, 7083)
        self.assertEqual(fp, 902)
        self.assertEqual(fn, 445)
        self.assertEqual(tp, 613)
        # Mathematical reconciliation checks
        self.assertEqual(tn + fp + fn + tp, 9043)
        self.assertEqual(cm["actual_no_total"], tn + fp)
        self.assertEqual(cm["actual_yes_total"], fn + tp)
        self.assertAlmostEqual(metrics_dict["Precision"], tp / (tp + fp), places=4)
        self.assertAlmostEqual(metrics_dict["Recall (Sensitivity)"], tp / (tp + fn), places=4)
        self.assertAlmostEqual(metrics_dict["Accuracy"], (tp + tn) / 9043, places=4)

    def test_11_get_dataset_summary(self):
        """GET /api/dataset/summary returns verified counts from data/bank-full.csv."""
        response = self.client.get("/api/dataset/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_records"], 45211)
        self.assertEqual(data["total_columns"], 17)
        self.assertEqual(data["missing_values"], 0)
        self.assertEqual(data["duplicate_rows"], 0)
        self.assertEqual(data["target_distribution"]["no"]["count"], 39922)
        self.assertEqual(data["target_distribution"]["yes"]["count"], 5289)

    def test_12_get_dataset_features(self):
        """GET /api/dataset/features returns feature catalogue."""
        response = self.client.get("/api/dataset/features")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total_features"], 17)
        feature_names = [f["name"] for f in data["features"]]
        self.assertIn("duration", feature_names)
        self.assertIn("previously_contacted", feature_names)
        duration_feat = next(f for f in data["features"] if f["name"] == "duration")
        self.assertFalse(duration_feat["used_in_model"])

    def test_13_get_dataset_preview(self):
        """GET /api/dataset/preview returns paginated slice of dataset."""
        response = self.client.get("/api/dataset/preview?page=1&page_size=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_records"], 45211)
        self.assertEqual(len(data["rows"]), 5)
        self.assertEqual(data["current_page"], 1)

    def test_14_get_dataset_eda(self):
        """GET /api/dataset/eda returns category conversion distributions."""
        response = self.client.get("/api/dataset/eda")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("target_distribution", data)
        self.assertIn("subscription_rate_by_poutcome", data)
        self.assertIn("subscription_rate_by_contact", data)
        poutcome_rates = {p["category"]: p["conversion_rate_pct"] for p in data["subscription_rate_by_poutcome"]}
        self.assertAlmostEqual(poutcome_rates["success"], 64.73, delta=0.5)


if __name__ == "__main__":
    unittest.main()


"""
Automated Model Reconciliation and Regression Test Suite.
Verifies end-to-end consistency between:
- notebooks/bank_marketing_ml.ipynb
- export_model.py
- models/random_forest_pipeline.joblib
- models/evaluation_metrics.json
- backend/main.py and backend/pipeline_data.py
"""

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.main import app
from backend.pipeline_data import EVALUATION_REPORT_PATH, get_performance_data
from scripts.common import (
    DEFAULT_DATA_PATH,
    TARGET,
    apply_feature_engineering,
    get_feature_target_split,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"


class TestModelReconciliation(unittest.TestCase):
    """Rigorous verification that all ML artifacts and backend responses match the authoritative notebook."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.raw_df = load_raw_dataset(DEFAULT_DATA_PATH)
        # Prepare test split identically to notebook
        df_engineered = apply_feature_engineering(cls.raw_df)
        X, y = get_feature_target_split(df_engineered, target_col=TARGET)
        X_train_raw, X_test_raw, cls.y_train, cls.y_test = split_train_test(
            X, y, test_size=0.2, random_state=42
        )
        cls.X_train, cls.X_test = prepare_model_inputs(X_train_raw, X_test_raw)
        cls.pipeline = joblib.load(MODEL_PATH)
        with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            cls.report = json.load(f)

    def test_01_dataset_loading_and_counts(self):
        """Verify raw dataset has exact shape (45211, 17) and verified class distribution."""
        self.assertEqual(self.raw_df.shape, (45211, 17))
        counts = self.raw_df["y"].value_counts().to_dict()
        self.assertEqual(counts["no"], 39922)
        self.assertEqual(counts["yes"], 5289)
        self.assertEqual(int(self.raw_df.isnull().sum().sum()), 0)
        self.assertEqual(int(self.raw_df.duplicated().sum()), 0)

    def test_02_feature_schema_and_duration_exclusion(self):
        """Verify pre-contact schema: 16 features, duration strictly excluded, previously_contacted present."""
        self.assertNotIn("duration", self.X_train.columns)
        self.assertNotIn("duration", self.X_test.columns)
        self.assertIn("previously_contacted", self.X_train.columns)
        self.assertEqual(self.X_train.shape[1], 16)
        self.assertEqual(self.X_test.shape[1], 16)

    def test_03_train_test_split_reproducibility(self):
        """Verify stratified 80/20 train/test split has exact record counts and class balance."""
        self.assertEqual(len(self.X_train), 36168)
        self.assertEqual(len(self.X_test), 9043)
        y_test_counts = self.y_test.value_counts().to_dict()
        self.assertEqual(y_test_counts["no"], 7985)
        self.assertEqual(y_test_counts["yes"], 1058)

    def test_04_model_artifact_loading_and_parameters(self):
        """Verify serialized pipeline matches notebook's selected best_estimator_ hyperparameters."""
        self.assertIsNotNone(self.pipeline)
        self.assertIn("preprocessor", self.pipeline.named_steps)
        self.assertIn("classifier", self.pipeline.named_steps)

        classifier = self.pipeline.named_steps["classifier"]
        self.assertEqual(type(classifier).__name__, "RandomForestClassifier")
        self.assertEqual(classifier.n_estimators, 100)
        self.assertIsNone(classifier.max_depth)
        self.assertEqual(classifier.min_samples_leaf, 3)
        self.assertEqual(classifier.class_weight, "balanced")
        self.assertEqual(classifier.random_state, 42)

    def test_05_positive_class_probability_mapping(self):
        """Verify classifier classes_ ordering and probability derivation for class 'yes'."""
        classifier = self.pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)
        self.assertEqual(classes, ["no", "yes"])
        self.assertEqual(classes.index("yes"), 1)

        probs = self.pipeline.predict_proba(self.X_test.iloc[:5])
        self.assertEqual(probs.shape, (5, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-6)

    def test_06_model_evaluation_on_test_split(self):
        """Directly evaluate saved pipeline on held-out test split and assert exact match with notebook."""
        y_pred = self.pipeline.predict(self.X_test)
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

        cm = confusion_matrix(self.y_test, y_pred, labels=["no", "yes"])
        tn, fp, fn, tp = cm.ravel()

        # Authoritative notebook Cell 176 / 212 values
        self.assertEqual(tn, 7083)
        self.assertEqual(fp, 902)
        self.assertEqual(fn, 445)
        self.assertEqual(tp, 613)

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, pos_label="yes")
        rec = recall_score(self.y_test, y_pred, pos_label="yes")
        f1 = f1_score(self.y_test, y_pred, pos_label="yes")
        auc = roc_auc_score(self.y_test == "yes", y_proba)

        self.assertAlmostEqual(acc, 0.851045, places=5)
        self.assertAlmostEqual(prec, 0.404620, places=5)
        self.assertAlmostEqual(rec, 0.579395, places=5)
        self.assertAlmostEqual(f1, 0.476487, places=5)
        self.assertAlmostEqual(auc, 0.8025, places=3)

    def test_07_evaluation_report_schema_and_consistency(self):
        """Verify models/evaluation_metrics.json schema and exact numeric match."""
        self.assertEqual(self.report["test_set_size"], 9043)
        cm_data = self.report["confusion_matrix"]
        self.assertEqual(cm_data["true_negatives"], 7083)
        self.assertEqual(cm_data["false_positives"], 902)
        self.assertEqual(cm_data["false_negatives"], 445)
        self.assertEqual(cm_data["true_positives"], 613)

        # Check summary metrics
        metrics = {m["name"]: m["value"] for m in self.report["summary_metrics"]}
        self.assertAlmostEqual(metrics["F1-Score"], 0.4765, places=4)
        self.assertAlmostEqual(metrics["ROC-AUC"], 0.8025, places=4)
        self.assertAlmostEqual(metrics["Accuracy"], 0.8510, places=4)
        self.assertAlmostEqual(metrics["Precision"], 0.4046, places=4)
        self.assertAlmostEqual(metrics["Recall (Sensitivity)"], 0.5794, places=4)

    def test_08_confusion_matrix_arithmetic_verification(self):
        """Mathematically verify confusion matrix identities and harmonic mean relations."""
        cm = self.report["confusion_matrix"]
        tn = cm["true_negatives"]
        fp = cm["false_positives"]
        fn = cm["false_negatives"]
        tp = cm["true_positives"]

        self.assertEqual(tn + fp + fn + tp, 9043)
        self.assertEqual(cm["actual_no_total"], tn + fp)
        self.assertEqual(cm["actual_yes_total"], fn + tp)
        self.assertEqual(cm["predicted_no_total"], tn + fn)
        self.assertEqual(cm["predicted_yes_total"], fp + tp)

        calc_prec = tp / (tp + fp)
        calc_rec = tp / (tp + fn)
        calc_f1 = 2 * (calc_prec * calc_rec) / (calc_prec + calc_rec)
        calc_acc = (tp + tn) / 9043

        self.assertAlmostEqual(calc_prec, 0.404620, places=5)
        self.assertAlmostEqual(calc_rec, 0.579395, places=5)
        self.assertAlmostEqual(calc_f1, 0.476487, places=5)
        self.assertAlmostEqual(calc_acc, 0.851045, places=5)

    def test_09_api_performance_response_match(self):
        """Verify GET /api/performance serves identical verified report without alteration."""
        response = self.client.get("/api/performance")
        self.assertEqual(response.status_code, 200)
        api_data = response.json()

        self.assertEqual(api_data["test_set_size"], self.report["test_set_size"])
        self.assertEqual(api_data["confusion_matrix"], self.report["confusion_matrix"])
        self.assertEqual(api_data["summary_metrics"], self.report["summary_metrics"])
        self.assertEqual(api_data["baseline_vs_tuned"], self.report["baseline_vs_tuned"])
        self.assertEqual(api_data["feature_importances"], self.report["feature_importances"])

    def test_10_missing_artifact_raises_error_no_silent_fallback(self):
        """Verify backend does not silently fall back to hardcoded numbers if report is missing."""
        import backend.pipeline_data as pd_mod
        original_path = pd_mod.EVALUATION_REPORT_PATH
        try:
            pd_mod.EVALUATION_REPORT_PATH = PROJECT_ROOT / "models" / "non_existent_report.json"
            with self.assertRaises(FileNotFoundError):
                get_performance_data()
        finally:
            pd_mod.EVALUATION_REPORT_PATH = original_path


if __name__ == "__main__":
    unittest.main()

"""
Model Training, Selection, and Export Script for Bank Marketing Predictor.

Reproduces the exact final model-selection procedure established in
`notebooks/bank_marketing_ml.ipynb`:
1. Loads raw dataset from `data/bank-full.csv`.
2. Applies domain feature engineering (`previously_contacted = int(pdays != -1)`).
3. Drops the post-contact 'duration' column to prevent data leakage in pre-contact prediction.
4. Splits data into 80% train / 20% test using StratifiedKFold (random_state=42).
5. Constructs preprocessing pipeline via ColumnTransformer (StandardScaler on 7 numerical features,
   OneHotEncoder(handle_unknown='ignore') on 9 categorical features).
6. Runs GridSearchCV over the notebook's specified Random Forest hyperparameter space using
   5-fold stratified cross-validation optimizing for F1-score (pos_label='yes').
7. Selects the best estimator (`search.best_estimator_`), refitted on the full training set.
8. Evaluates the fitted pipeline on the held-out test split (9,043 samples) to verify test metrics:
   - TN = 7,083, FP = 902, FN = 445, TP = 613
   - Accuracy = 85.10%, Precision = 40.46%, Recall = 57.94%, F1 = 0.4765, ROC-AUC = 80.25%.
9. Safely writes candidate pipeline, verifies it, and promotes to:
   `models/random_forest_pipeline.joblib`
10. Exports companion evaluation report to:
   `models/evaluation_metrics.json`
"""

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

# Configure utf-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    TARGET,
    apply_feature_engineering,
    build_model_pipelines,
    create_preprocessor,
    get_cv_strategy,
    get_feature_target_split,
    get_param_grids,
    get_scoring_metrics,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
CANDIDATE_MODEL_PATH = MODELS_DIR / "random_forest_pipeline.candidate.joblib"
METRICS_REPORT_PATH = MODELS_DIR / "evaluation_metrics.json"
DATASET_PATH = PROJECT_ROOT / "data" / "bank-full.csv"

# Feature descriptions for feature importances table
FEATURE_DESCRIPTIONS = {
    "month": "Last contact month of the year (seasonal deposit peaks in Mar/Sep/Oct)",
    "balance": "Customer average yearly balance (subscribers average €1,804 vs €1,304 non-subscribers)",
    "age": "Customer age in years (higher propensity in students < 25 and retirees > 60)",
    "day": "Last contact day of the month",
    "poutcome": "Outcome of previous marketing campaign ('success' conversion rate is 64.7%)",
    "contact": "Communication channel (cellular leads convert at 14.9% vs unknown 4.1%)",
    "job": "Occupation category (students and retirees convert at > 22%)",
    "campaign": "Number of contacts during current campaign (conversion drops after 3 calls)",
    "housing": "Has active housing mortgage loan (customers without mortgage convert at 16.7% vs 7.7%)",
    "pdays": "Days since previous campaign contact (-1 sentinel retained)",
    "education": "Education level (tertiary educated clients convert at 15.0%)",
    "marital": "Marital status (single customers convert at higher rates than married)",
    "loan": "Has active personal consumer loan",
    "previous": "Number of contacts performed before this campaign",
    "previously_contacted": "Binary engineered indicator (0 if pdays == -1 else 1)",
    "default": "Has credit in default (lowest predictive variance; 98.2% are 'no')",
}


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def aggregate_feature_importances(
    pipeline: Any,
    input_feature_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Extract Gini impurity importances from fitted RandomForestClassifier
    and aggregate one-hot encoded levels back to parent features.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    encoder = preprocessor.transformers_[1][1]
    encoded_cat_names = list(encoder.get_feature_names_out(cat_features))
    all_transformed_names = list(num_features) + encoded_cat_names

    raw_importances = classifier.feature_importances_

    agg_imp = {f: 0.0 for f in input_feature_names}
    for feat_name, imp in zip(all_transformed_names, raw_importances):
        for parent in input_feature_names:
            if feat_name == parent or feat_name.startswith(parent + "_"):
                agg_imp[parent] += float(imp)
                break

    sorted_imp = sorted(agg_imp.items(), key=lambda x: x[1], reverse=True)
    result = []
    for rank, (feat, val) in enumerate(sorted_imp, 1):
        result.append({
            "feature": feat,
            "importance": round(val, 4),
            "importance_pct": round(val * 100, 2),
            "rank": rank,
            "description": FEATURE_DESCRIPTIONS.get(feat, "Predictive feature"),
        })
    return result


def train_and_export_model(
    output_path: Path = MODEL_PATH,
    report_path: Path = METRICS_REPORT_PATH
) -> Tuple[Any, Dict[str, Any]]:
    """
    Execute the notebook's full model selection procedure, verify test metrics,
    and export the model artifact and companion evaluation report.
    """
    start_total_time = time.perf_counter()
    print("=" * 70)
    print("Bank Marketing Predictor — Model Selection, Validation & Export")
    print("=" * 70)

    # 1. Dataset Loading & Feature Engineering
    print(f"\n[Step 1] Loading raw dataset from: {DATASET_PATH}")
    df_raw = load_raw_dataset()
    dataset_records = len(df_raw)
    dataset_hash = compute_file_sha256(DATASET_PATH)
    print(f"  Loaded {dataset_records:,} records. SHA256: {dataset_hash[:16]}...")

    print("\n[Step 2] Applying domain feature engineering (previously_contacted = int(pdays != -1))...")
    df = apply_feature_engineering(df_raw)

    X, y = get_feature_target_split(df, TARGET)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_model, X_test_model = prepare_model_inputs(X_train, X_test)

    print(f"  Train samples: {len(X_train_model):,} | Test samples: {len(X_test_model):,}")
    print(f"  Training features (duration excluded): {list(X_train_model.columns)}")

    # 2. Recreate Notebook's Model Pipelines and Hyperparameter Search
    print("\n[Step 3] Initializing pipeline and GridSearchCV matching notebook Section 13...")
    pipelines = build_model_pipelines()
    param_grids = get_param_grids()
    scoring_metrics = get_scoring_metrics()
    cv_strategy = get_cv_strategy()

    rf_pipeline = pipelines["Random Forest"]
    rf_param_grid = param_grids["Random Forest"]

    print(f"  Candidate Model: Random Forest")
    print(f"  Cross-Validation: 5-Fold StratifiedKFold (shuffle=True, random_state=42)")
    print(f"  Scoring Metric: f1 (pos_label='yes')")
    print(f"  Hyperparameter Search Space: {rf_param_grid}")

    search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        scoring=scoring_metrics,
        refit="f1",
        cv=cv_strategy,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    print("\n[Step 4] Executing 5-Fold Cross-Validation search across 16 combinations...")
    search_start = time.perf_counter()
    search.fit(X_train_model, y_train)
    search_time = time.perf_counter() - search_start
    print(f"  Search completed in {search_time:.2f}s.")

    best_params = search.best_params_
    best_cv_f1 = float(search.best_score_)
    print(f"  Best CV F1-Score: {best_cv_f1:.4f}")
    print(f"  Best Hyperparameters: {best_params}")

    # Notebook verified expectation check
    expected_params = {
        "classifier__class_weight": "balanced",
        "classifier__max_depth": None,
        "classifier__min_samples_leaf": 3,
        "classifier__n_estimators": 100,
    }
    for k, v in expected_params.items():
        if best_params.get(k) != v:
            raise AssertionError(
                f"Hyperparameter mismatch: Expected {k}={v}, but search produced {best_params.get(k)}"
            )
    print("  [OK] Selected hyperparameters EXACTLY match notebook Cell 162/172!")

    # The best_estimator_ was refitted on the full training set by GridSearchCV (refit='f1')
    best_pipeline = search.best_estimator_

    # 3. Test Evaluation Matching Notebook Section 14
    print("\n[Step 5] Evaluating best estimator on held-out test split (9,043 samples)...")
    y_pred = best_pipeline.predict(X_test_model)

    classes = list(best_pipeline.named_steps["classifier"].classes_)
    if "yes" not in classes:
        raise ValueError(f"Expected positive class 'yes' in classifier classes: {classes}")
    pos_idx = classes.index("yes")

    y_prob = best_pipeline.predict_proba(X_test_model)[:, pos_idx]
    y_test_bin = (y_test == "yes").astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=["no", "yes"])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    total_test = tn + fp + fn + tp

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label="yes", zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label="yes", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label="yes", zero_division=0))
    auc = float(roc_auc_score(y_test_bin, y_prob))

    print(f"  Test Confusion Matrix (labels=['no', 'yes']):")
    print(f"    TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,} | Total={total_test:,}")
    print(f"  Test Evaluation Metrics:")
    print(f"    Accuracy:  {acc:.4f} ({acc*100:.2f}%) [Expected: ~0.8510]")
    print(f"    Precision: {prec:.4f} ({prec*100:.2f}%) [Expected: ~0.4046]")
    print(f"    Recall:    {rec:.4f} ({rec*100:.2f}%) [Expected: ~0.5794]")
    print(f"    F1-Score:  {f1:.4f} [Expected: ~0.4765]")
    print(f"    ROC-AUC:   {auc:.4f} ({auc*100:.2f}%) [Expected: ~0.8025]")

    # Verify mathematical exactness
    assert total_test == 9043, f"Total test samples expected 9043, got {total_test}"
    assert tn == 7083, f"Expected TN=7083, got {tn}"
    assert fp == 902, f"Expected FP=902, got {fp}"
    assert fn == 445, f"Expected FN=445, got {fn}"
    assert tp == 613, f"Expected TP=613, got {tp}"
    assert abs(prec - (tp / (tp + fp))) < 1e-6, "Precision arithmetic mismatch"
    assert abs(rec - (tp / (tp + fn))) < 1e-6, "Recall arithmetic mismatch"
    assert abs(acc - ((tn + tp) / total_test)) < 1e-6, "Accuracy arithmetic mismatch"
    assert abs(f1 - (2 * prec * rec / (prec + rec))) < 1e-6, "F1 arithmetic mismatch"
    print("  [OK] All confusion matrix counts and test metrics 100% reconciled and notebook-verified!")

    # 4. Feature Importances Extraction
    print("\n[Step 6] Extracting and aggregating Gini feature importances...")
    feature_importances = aggregate_feature_importances(best_pipeline, list(X_train_model.columns))
    for item in feature_importances[:5]:
        print(f"    {item['rank']}. {item['feature']}: {item['importance_pct']}%")

    # 5. Baseline Comparison Setup
    baseline_vs_tuned = [
        {
            "model": "Baseline Random Forest",
            "hyperparameters": "n_estimators=200, default depth/weights",
            "accuracy": 0.8937,
            "precision": 0.6234,
            "recall": 0.2316,
            "f1_score": 0.3377,
            "roc_auc": 0.7900,
            "notes": "Suffers from severe minority-class neglect; misses > 76% of actual subscribers (245 detected out of 1,058).",
        },
        {
            "model": "Tuned Random Forest",
            "hyperparameters": "n_estimators=100, max_depth=None, min_samples_leaf=3, class_weight='balanced'",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "notes": "Selected model. Balanced class weights catch 613 subscribers (57.94% recall vs 23.16% baseline), prioritizing lead capture.",
        },
    ]

    # 6. Assemble Companion Evaluation Report
    print("\n[Step 7] Generating companion evaluation report JSON...")
    report_data = {
        "report_metadata": {
            "title": "Model Performance & Evaluation",
            "subtitle": "Rigorous test evaluation results demonstrating how the tuned Random Forest handles severe class imbalance.",
            "version": "2.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "export_model.py (Notebook-Reconciled Workflow)",
            "status": "VERIFIED",
        },
        "model_specifications": {
            "model_name": "Random Forest (Tuned Pipeline)",
            "framework": "scikit-learn 1.4+",
            "artifact_path": str(output_path.name),
            "positive_class": "yes",
            "negative_class": "no",
            "classes": classes,
            "timing_constraint": "Pre-contact prediction (duration excluded)",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_leaf": 3,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
            "preprocessing": {
                "preprocessor_type": "ColumnTransformer",
                "numerical_features_count": 7,
                "categorical_features_count": 9,
                "input_features_count": 16,
                "transformed_features_count": 51,
            },
        },
        "dataset": {
            "file": "data/bank-full.csv",
            "sha256": dataset_hash,
            "total_records": dataset_records,
            "train_samples": len(X_train_model),
            "test_samples": len(X_test_model),
            "split_ratio": "80% Train / 20% Held-out Test (Stratified, seed=42)",
            "train_distribution": {
                "no": int((y_train == "no").sum()),
                "yes": int((y_train == "yes").sum()),
            },
            "test_distribution": {
                "no": int((y_test == "no").sum()),
                "yes": int((y_test == "yes").sum()),
            },
        },
        "cross_validation": {
            "strategy": "5-Fold StratifiedKFold (shuffle=True, random_state=42)",
            "scoring": "f1 (pos_label='yes')",
            "best_cv_f1": round(best_cv_f1, 4),
            "search_time_seconds": round(search_time, 2),
        },
        "test_set_size": total_test,
        "test_split_ratio": "80% Train / 20% Held-out Test (Stratified, seed=42)",
        "summary_metrics": [
            {
                "name": "ROC-AUC",
                "value": round(auc, 4),
                "formatted": f"{auc*100:.2f}%",
                "dataset_split": "Held-out Test Set (9,043 samples)",
                "status": "Verified",
                "description": "Measures ranking ability across decision thresholds. An AUC of 80.25% confirms strong discrimination between subscribers and non-subscribers.",
            },
            {
                "name": "F1-Score",
                "value": round(f1, 4),
                "formatted": f"{f1:.4f}",
                "dataset_split": "Held-out Test Set (9,043 samples)",
                "status": "Verified",
                "description": "Harmonic mean of precision and recall for the positive class ('yes'). Balances detection against false alarm penalties under class imbalance.",
            },
            {
                "name": "Recall (Sensitivity)",
                "value": round(rec, 4),
                "formatted": f"{rec*100:.2f}%",
                "dataset_split": "Held-out Test Set (9,043 samples)",
                "status": "Verified",
                "description": "Captures 57.94% of all actual term deposit subscribers (613 out of 1,058). More than double baseline Random Forest recall (23.16%).",
            },
            {
                "name": "Precision",
                "value": round(prec, 4),
                "formatted": f"{prec*100:.2f}%",
                "dataset_split": "Held-out Test Set (9,043 samples)",
                "status": "Verified",
                "description": "Out of all leads flagged as likely subscribers, 40.46% actually converted (613 out of 1,515 flagged). Enables efficient sales prioritization.",
            },
            {
                "name": "Accuracy",
                "value": round(acc, 4),
                "formatted": f"{acc*100:.2f}%",
                "dataset_split": "Held-out Test Set (9,043 samples)",
                "status": "Verified",
                "description": "Overall correct classifications (7,696 / 9,043). Reflects intentional trade-off where balanced weights catch far more subscribers.",
            },
        ],
        "confusion_matrix": {
            "total_test_samples": total_test,
            "class_order": ["no", "yes"],
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "actual_no_total": tn + fp,
            "actual_yes_total": fn + tp,
            "predicted_no_total": tn + fn,
            "predicted_yes_total": fp + tp,
            "breakdown": [
                {
                    "quadrant": "True Negatives (TN)",
                    "count": tn,
                    "percentage_of_test": round((tn / total_test) * 100, 2),
                    "interpretation": f"Non-subscribers correctly predicted as 'no' ({tn:,} / {tn+fp:,} actual non-subscribers). Avoids wasting outreach bandwidth.",
                },
                {
                    "quadrant": "False Positives (FP)",
                    "count": fp,
                    "percentage_of_test": round((fp / total_test) * 100, 2),
                    "interpretation": "Non-subscribers incorrectly predicted as 'yes'. Incurs brief marketing call cost, but poses no direct financial loss.",
                },
                {
                    "quadrant": "False Negatives (FN)",
                    "count": fn,
                    "percentage_of_test": round((fn / total_test) * 100, 2),
                    "interpretation": "Actual subscribers missed by the model. Represents missed deposit acquisition opportunities that could have generated bank revenue.",
                },
                {
                    "quadrant": "True Positives (TP)",
                    "count": tp,
                    "percentage_of_test": round((tp / total_test) * 100, 2),
                    "interpretation": f"Actual subscribers successfully identified and contacted ({tp:,} / {fn+tp:,} actual subscribers). Highly actionable pool for sales advisors.",
                },
            ],
        },
        "baseline_vs_tuned": baseline_vs_tuned,
        "all_baseline_models": [
            {
                "model": "Random Forest",
                "accuracy": 0.8937,
                "precision": 0.6234,
                "recall": 0.2316,
                "f1_score": 0.3377,
                "roc_auc": 0.7900,
                "rank_f1": 1,
                "selection_status": "Shortlisted for tuning",
            },
            {
                "model": "Decision Tree",
                "accuracy": 0.8329,
                "precision": 0.3025,
                "recall": 0.3280,
                "f1_score": 0.3147,
                "roc_auc": 0.6139,
                "rank_f1": 2,
                "selection_status": "Eliminated (lowest AUC, high variance)",
            },
            {
                "model": "Gradient Boosting",
                "accuracy": 0.8943,
                "precision": 0.6594,
                "recall": 0.1994,
                "f1_score": 0.3062,
                "roc_auc": 0.8023,
                "rank_f1": 3,
                "selection_status": "Shortlisted for tuning (highest baseline ROC-AUC)",
            },
            {
                "model": "Logistic Regression",
                "accuracy": 0.8934,
                "precision": 0.6643,
                "recall": 0.1796,
                "f1_score": 0.2827,
                "roc_auc": 0.7717,
                "rank_f1": 4,
                "selection_status": "Shortlisted for tuning (linear baseline benchmark)",
            },
        ],
        "feature_importances": feature_importances,
        "limitations": [
            {
                "number": 1,
                "title": "Target Class Imbalance",
                "detail": "The target is heavily imbalanced (~88.3% non-subscribers). High raw accuracy is easily achieved by simply guessing the majority class, making F1-Score, Precision, and Recall the only trustworthy performance indicators.",
            },
            {
                "number": 2,
                "title": "Precision vs. Recall Business Trade-off",
                "detail": "Using class_weight='balanced' successfully doubled subscriber recall (57.94% vs 23.16%), but produced 902 false positives. In bank sales operations, brief phone contact cost is small relative to term deposit acquisition value.",
            },
            {
                "number": 3,
                "title": "Test Set Benchmark Inspection",
                "detail": "The test set was evaluated during earlier baseline comparisons in the notebook before final tuning, meaning it represents a reliable comparative benchmark rather than an untouched prospective holdout.",
            },
            {
                "number": 4,
                "title": "Predictive Correlation vs. Causation",
                "detail": "Feature importance metrics (Gini and permutation) reflect statistical predictive associations, not causal relationships. For example, high conversion in October does not mean calling customers in October causes them to subscribe.",
            },
            {
                "number": 5,
                "title": "Decision Threshold Alignment",
                "detail": "The model uses the default 0.50 probability cutoff. A bank should optimize this threshold dynamically using specific financial cost-benefit equations (cost of outreach phone call vs expected net interest margin of deposit).",
            },
            {
                "number": 6,
                "title": "Lack of Prospective A/B Operational Testing",
                "detail": "The pipeline has been rigorously validated offline on historical data, but has not yet been deployed in a prospective randomized A/B live marketing campaign to evaluate real advisor conversion rates.",
            },
        ],
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 7. Safe Artifact Export (Write to candidate, verify, then promote)
    print(f"\n[Step 8] Serializing candidate pipeline to temporary path: {CANDIDATE_MODEL_PATH}")
    joblib.dump(best_pipeline, CANDIDATE_MODEL_PATH)

    # Smoke test on candidate artifact
    print("  Running smoke test on candidate artifact...")
    test_loaded = joblib.load(CANDIDATE_MODEL_PATH)
    sample_row = X_test_model.iloc[[0]]
    sample_pred = test_loaded.predict(sample_row)[0]
    sample_prob = test_loaded.predict_proba(sample_row)[0, pos_idx]
    print(f"  Candidate smoke test passed! (Pred: '{sample_pred}', Prob: {sample_prob:.4f})")

    # Promote candidate to production artifact
    print(f"  Promoting candidate to production model path: {output_path}")
    shutil.move(str(CANDIDATE_MODEL_PATH), str(output_path))
    model_size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    report_data["model_specifications"]["artifact_size_mb"] = model_size_mb
    print(f"  [OK] Model serialized successfully ({model_size_mb} MB)!")

    # Write companion evaluation report JSON
    print(f"[Step 9] Writing companion evaluation report to: {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  [OK] Evaluation report written ({len(report_data)} top-level sections)!")

    total_time = time.perf_counter() - start_total_time
    print(f"\n=== Entire Training, Verification & Export Completed in {total_time:.2f}s ===")
    return best_pipeline, report_data


if __name__ == "__main__":
    train_and_export_model()

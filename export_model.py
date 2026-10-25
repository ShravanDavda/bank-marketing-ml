"""
Model Export Script for Bank Marketing Predictor.

Reproduces and exports the exact final tuned Random Forest pipeline established in
`notebooks/bank_marketing_ml.ipynb` and `scripts/14_final_model_evaluation.py`.

The saved artifact contains the complete end-to-end pipeline:
1. Preprocessor: ColumnTransformer (StandardScaler on 7 numerical features,
   OneHotEncoder(handle_unknown='ignore') on 9 categorical features).
2. Classifier: Tuned RandomForestClassifier(
       n_estimators=200,
       max_depth=15,
       min_samples_leaf=3,
       class_weight='balanced',
       random_state=42,
       n_jobs=-1
   ).

The contact 'duration' column is strictly excluded from predictive inputs.
Artifact output: models/random_forest_pipeline.joblib
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    TARGET,
    apply_feature_engineering,
    create_preprocessor,
    get_feature_target_split,
    load_raw_dataset,
    prepare_model_inputs,
    split_train_test,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"


def build_final_pipeline() -> Pipeline:
    """
    Construct the final model pipeline reusing Section 9 and 14 configurations.
    """
    preprocessor = create_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_export_model(output_path: Path = MODEL_PATH) -> Pipeline:
    """
    Fit the exact final pipeline on training data and serialize to disk.
    """
    print(f"Loading raw dataset from data/bank-full.csv...")
    df_raw = load_raw_dataset()

    print("Applying Section 7 feature engineering (previously_contacted)...")
    df = apply_feature_engineering(df_raw)

    X, y = get_feature_target_split(df, TARGET)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_model, X_test_model = prepare_model_inputs(X_train, X_test)

    print(f"Training input shape: {X_train_model.shape} (duration excluded)")
    print(f"Fitting final tuned Random Forest pipeline...")
    pipeline = build_final_pipeline()
    pipeline.fit(X_train_model, y_train)

    # Verification on held-out test data
    y_pred = pipeline.predict(X_test_model)
    classes = list(pipeline.named_steps["classifier"].classes_)
    pos_idx = classes.index("yes")
    y_prob = pipeline.predict_proba(X_test_model)[:, pos_idx]
    y_test_bin = (y_test == "yes").astype(int)

    test_f1 = f1_score(y_test, y_pred, pos_label="yes", zero_division=0)
    test_roc_auc = roc_auc_score(y_test_bin, y_prob)

    print(f"Model validation metrics on held-out test set:")
    print(f"  - Classes: {classes} (Positive: 'yes' at index {pos_idx})")
    print(f"  - Test F1-Score: {test_f1:.4f} (Expected ~0.4797)")
    print(f"  - Test ROC-AUC : {test_roc_auc:.4f} (Expected ~0.8010)")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Pipeline successfully serialized to: {output_path}")

    # Smoke test on loaded artifact
    loaded_pipeline = joblib.load(output_path)
    sample_input = X_test_model.iloc[[0]]
    sample_pred = loaded_pipeline.predict(sample_input)[0]
    sample_prob = loaded_pipeline.predict_proba(sample_input)[0, pos_idx]

    print(f"Smoke test on loaded pipeline:")
    print(f"  - Sample prediction: {sample_pred}")
    print(f"  - Sample probability ('yes'): {sample_prob:.4f}")

    return loaded_pipeline


if __name__ == "__main__":
    train_and_export_model()

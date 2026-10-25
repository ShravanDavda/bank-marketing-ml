# Bank Marketing ML — Script Pipeline

This folder (`scripts/`) contains an organized, modular, section-wise conversion of the completed machine learning notebook located at `notebooks/bank_marketing_ml.ipynb`.

The original notebook remains the untouched academic source of truth. These Python scripts provide a clean, testable, and reusable workflow suitable for automated testing, batch processing, and integration with a prediction interface (e.g., Streamlit).

---

## 1. Folder Architecture and Notebook Mapping

| Script Name | Notebook Section | Standalone? | Description |
| :--- | :--- | :---: | :--- |
| `common.py` | Cross-section | Yes | Centralized configuration, column names, dataset loaders, preprocessor builders, and metric definitions. |
| `01_imports.py` | Section 1 & 2 | Yes | Problem definition docstring, environment diagnostics, and dependency verification. |
| `02_load_dataset.py` | Section 3 | Yes | Loads `data/bank-full.csv` (semicolon-separated) and verifies dimensions (45,211 × 17). |
| `03_data_understanding.py` | Section 4 | Yes | Analyzes dimensions, column data types, missing values, duplicate rows, and target distribution. |
| `04_data_cleaning.py` | Section 5 | Yes | Audits data quality, categorical values, numerical ranges, and text whitespace. Confirms no row drops. |
| `05_eda.py` | Section 6 | Yes (Interactive) | Plots target imbalance, numerical histograms, categorical countplots, subscription conversion rates, and boxplots. |
| `06_feature_engineering.py` | Section 7 | Yes | Creates the binary indicator `previously_contacted` (`0` if `pdays == -1`, else `1`) and validates assertions. |
| `07_train_test_split.py` | Section 8 | Yes | Executes 80:20 stratified split (`random_state=42`) and verifies index disjunction and class stratification. |
| `08_preprocessing.py` | Section 9 | Yes | Drops `duration`, creates `ColumnTransformer` (StandardScaler + OneHotEncoder), and verifies finite matrices. |
| `09_baseline_training.py` | Section 10 | Yes (Trains) | Fits unified pipelines for Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. |
| `10_model_evaluation.py` | Section 11 | Yes (Trains/Plots) | Evaluates baseline models on test set (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC curves). |
| `11_model_comparison.py` | Section 12 | Yes (Trains/Plots) | Ranks models by F1-Score, plots comparative bar charts, and identifies candidates for cross-validation. |
| `12_cross_validation.py` | Section 13 (13.1–13.3) | Yes | Configures 5-fold stratified CV, scoring metrics dictionary, and candidate hyperparameter grids. |
| `13_hyperparameter_tuning.py` | Section 13 (13.4–13.8) | Yes (Tunes) | Runs `GridSearchCV` on candidate models with `refit="f1"` and stores the refitted tuned pipelines. |
| `14_final_model_evaluation.py` | Section 14 | Yes (Trains/Plots) | Evaluates final tuned Random Forest vs. baseline Random Forest, extracts confusion counts, and displays matrix. |
| `15_model_interpretation.py` | Section 15 | Yes (Trains/Plots) | Extracts built-in Gini importance (raw and aggregated) and computes test permutation importance (F1 drop). |
| `16_conclusion.py` | Section 16 | Yes | Synthesizes project metadata, final test metrics, confusion matrix breakdown, limitations, and future work. |

---

## 2. Dataset Path Assumptions

All scripts compute file paths dynamically relative to `PROJECT_ROOT` using Python's `pathlib.Path`:

```text
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bank-full.csv"
```

You can execute any script from any working directory (the project root, `scripts/`, or externally) without broken relative path errors. The original dataset in `data/bank-full.csv` is accessed in read-only mode and is never modified.

---

## 3. Required Dependencies

The scripts use the dependencies declared in `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
joblib
```

To run using the local virtual environment:

```powershell
.\.venv\Scripts\python scripts/01_imports.py
```

---

## 4. How to Run the Scripts

Each script is equipped with an entry-point guard (`if __name__ == "__main__": main()`).

### Fast Verification & Inspection Scripts
These scripts execute quickly without heavy model training:

```bash
python scripts/01_imports.py
python scripts/02_load_dataset.py
python scripts/03_data_understanding.py
python scripts/04_data_cleaning.py
python scripts/06_feature_engineering.py
python scripts/07_train_test_split.py
python scripts/08_preprocessing.py
python scripts/12_cross_validation.py
python scripts/16_conclusion.py
```

### Exploratory & Visualization Scripts
These scripts generate interactive Matplotlib/Seaborn plots:

```bash
python scripts/05_eda.py
```

### Model Training & Evaluation Scripts
These scripts train baseline pipelines and evaluate them on held-out test data:

```bash
python scripts/09_baseline_training.py
python scripts/10_model_evaluation.py
python scripts/11_model_comparison.py
python scripts/14_final_model_evaluation.py
python scripts/15_model_interpretation.py
```

### Hyperparameter Tuning Script
Runs 5-fold cross-validation grid search (defaults to parallel execution `n_jobs=-1` for candidate Random Forest):

```bash
python scripts/13_hyperparameter_tuning.py
```

---

## 5. Model Artifacts and Future UI Integration Status

- **Status of Saved Artifact**: The original repository and notebook did not contain a serialized model file (`.joblib` or `.pkl`).
- **Reproducibility**: The exact final pipeline (preprocessing `ColumnTransformer` + tuned `RandomForestClassifier`) can be reliably constructed and exported using `common.py` or `scripts/14_final_model_evaluation.py`.
- **Pre-contact Constraint**: All scripts strictly exclude the `duration` column from feature inputs (`X_model = X.drop(columns=["duration"])`), ensuring compatibility with a real-world pre-campaign prediction application.

---

## 6. Known Limitations

1. **Target Imbalance**: The dataset has ~88.3% non-subscribers and ~11.7% subscribers. Evaluation prioritizes F1-Score, Precision, Recall, and ROC-AUC over raw Accuracy.
2. **Precision-Recall Trade-off**: The tuned model substantially improves recall (detecting subscribers) from ~23% to ~54%, but reduces precision due to increased false positives.
3. **Correlation vs. Causation**: Feature importance rankings indicate predictive correlation and do not imply causal mechanisms for customer behavior.

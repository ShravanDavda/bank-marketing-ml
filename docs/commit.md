# Bank Marketing ML — GitHub Commit & Collaboration Plan

## 1. Project Overview

**Project:** Bank Marketing Campaign — Customer Subscription Prediction  
**Repository:** `bank-marketing-ml`  
**Primary branch:** `main`  
**Notebook:** `notebooks/bank_marketing_ml.ipynb`  
**Dataset:** `data/bank-full.csv` (local file, excluded from Git)

This document defines the repository setup, contribution responsibilities, notebook section ownership, commit messages, and final verification process for all five members.

## 2. Repository Structure

```text
bank-marketing-ml/
├── data/
│   └── bank-full.csv                 # Local only; not committed
├── notebooks/
│   └── bank_marketing_ml.ipynb       # Main ML notebook
├── docs/
│   ├── GITHUB_COMMIT_PLAN.md
│   ├── 02-data/
│   ├── 03-preprocessing/
│   ├── 04-modeling/
│   ├── 05-interpretation/
│   └── 06-reporting/
├── experiments/                      # Optional future experiments
├── .gitignore
├── README.md
└── requirements.txt
```

The dataset must be obtained separately by each member. The README should explain its source and local placement.

## 3. Initial Repository Setup

The repository coordinator performs the initial setup.

### 3.1 Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/bank-marketing-ml.git
cd bank-marketing-ml
```

If the repository is already cloned, open the existing folder instead.

### 3.2 Verify the repository

```bash
git status
git branch --show-current
git remote -v
```

Confirm that the current branch is `main` and the remote points to the intended GitHub repository.

### 3.3 Configure Git identity

Each member should configure their own identity:

```bash
git config user.name "Your Full Name"
git config user.email "your-github-email@example.com"
```

### 3.4 Configure `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/

# Virtual environments
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# Local dataset
data/*
!data/.gitkeep

# Secrets
.env
.env.*
!.env.example

# OS/editor files
.DS_Store
Thumbs.db
.vscode/
.idea/
```

Verify that the dataset is ignored:

```bash
git check-ignore data/bank-full.csv
```

Do not commit the raw dataset unless its redistribution terms and repository requirements have been checked.

### 3.5 Initial commit

Add the initial repository files, such as `.gitignore`, `README.md`, `requirements.txt`, and the notebook if it is ready.

```bash
git add .gitignore README.md requirements.txt notebooks/
git status
git commit -m "chore: initialize bank marketing ML project"
git push -u origin main
```

Review the staged files before committing. If the initial setup is already committed, do not create a duplicate commit. If GitHub has a separate initial commit, reconcile the histories rather than force-pushing.

---

## 4. Member Responsibilities

| Member | Notebook sections | Responsibility |
|---|---|---|
| Member 1 | Sections 1–4 and 6 | Problem definition, imports, dataset loading, data understanding and EDA |
| Member 2 | Sections 5, 7–9 | Data cleaning, feature engineering, train-test split and preprocessing |
| Member 3 | Sections 10–12 | Baseline model training, evaluation and comparison |
| Member 4 | Section 13 | Cross-validation and hyperparameter tuning |
| Member 5 | Sections 14–16 | Final model evaluation, interpretation, conclusion and future work |

**Shared notebook rule:** All members edit the same notebook, so only one member should edit it at a time. The next member starts after the previous member's work has been committed and synchronized.

## 5. Section-Wise Commit Plan

Commit messages should describe the actual work performed. The following messages are recommended milestones, not a requirement to make a commit when no meaningful change exists.

### Member 1 — Problem Definition, Dataset & EDA

**Sections 1–4 and 6**

| Section | Scope | Commit message |
|---|---|---|
| 1 | Problem definition and project objective | `docs: define bank marketing ML project objective` |
| 2 | Import libraries | `chore: add data analysis library imports` |
| 3 | Dataset loading | `feat: add bank marketing dataset loading` |
| 4 | Data understanding and summaries | `feat: add dataset understanding and summary analysis` |
| 6 | EDA visualizations and findings | `feat: add exploratory data analysis` |

The final combined commit, if the sections are completed together, can be:

`feat: add project introduction, data understanding, and EDA`

**Member 1 must not implement Section 5**, which belongs to Member 2.

### Member 2 — Data Cleaning & Preprocessing

**Sections 5, 7, 8 and 9**

| Section | Scope | Commit message |
|---|---|---|
| 5 | Data quality checks and cleaning decisions | `feat: add data cleaning and quality checks` |
| 7 | Feature engineering and target separation | `feat: add feature engineering and target separation` |
| 8 | Stratified train-test split | `feat: add stratified train test split` |
| 9 | Encoding, scaling and preprocessing pipeline | `feat: add data preprocessing pipeline` |

The final combined commit, if all sections are completed together, can be:

`feat: add data cleaning and preprocessing pipeline`

Important implementation rules:

- Preserve the original dataset for reference.
- Do not automatically treat `"unknown"` as missing.
- Do not delete valid negative balances.
- Interpret `pdays = -1` according to its dataset meaning.
- Exclude `duration` from model inputs because it is known only after contact.
- Split before fitting preprocessing transformations.
- Use a reproducible split with `random_state=42` and `stratify=y`.

### Member 3 — Baseline Models & Comparison

**Sections 10–12**

| Section | Scope | Commit message |
|---|---|---|
| 10 | Train four baseline models | `feat: add baseline classification models` |
| 11 | Calculate evaluation metrics and confusion matrices | `feat: add baseline model evaluation metrics` |
| 12 | Compare the baseline models | `feat: add baseline model comparison` |

The final combined commit, if completed together, can be:

`feat: add baseline model training and comparison`

Models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Use the preprocessing pipeline from Section 9. Calculate precision, recall and F1-score with `"yes"` as the positive class. Calculate ROC-AUC using predicted probabilities or decision scores.

These are baseline results. Hyperparameter tuning belongs to Member 4.

### Member 4 — Cross-Validation & Hyperparameter Tuning

**Section 13**

| Subsection | Scope | Commit message |
|---|---|---|
| 13.1 | Stratified five-fold cross-validation | `feat: add stratified cross-validation strategy` |
| 13.2 | Multi-metric scoring | `feat: configure classification scoring metrics` |
| 13.3 | Hyperparameter grids | `feat: define model hyperparameter grids` |
| 13.4 | GridSearchCV execution | `feat: add cross-validated hyperparameter tuning` |
| 13.5 | Tuned model comparison | `feat: compare tuned model results` |

The final combined commit can be:

`feat: add cross-validation and hyperparameter tuning`

Implementation requirements:

- Tune Logistic Regression, Random Forest and Gradient Boosting.
- Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Keep preprocessing inside the pipeline passed to GridSearchCV.
- Use `refit="f1"` and the appropriate positive-class scorers.
- Use `n_jobs=1` for controlled resource usage.
- Do not use the test set to select parameters.

The established grids contain 30 configurations and require up to 150 model fits across five folds.

### Member 5 — Final Evaluation & Interpretation

**Sections 14–16**

| Section | Scope | Commit message |
|---|---|---|
| 14 | Final model selection and held-out evaluation | `feat: add final model evaluation` |
| 15 | Built-in and permutation feature importance | `feat: add model interpretation and feature importance` |
| 16 | Conclusion, limitations and future work | `docs: add project conclusion and future work` |

The final combined commit can be:

`feat: add final model evaluation and interpretation`

Implementation requirements:

- Use actual results from Section 13.
- Evaluate the selected model using the held-out test set.
- Report accuracy, precision, recall, F1-score, ROC-AUC and the confusion matrix.
- Exclude `duration` from prediction features and feature-importance analysis.
- Explain that feature importance is not proof of causation.
- Disclose that the test set was inspected during development and is not a completely untouched final benchmark.

---

## 6. Git Workflow for Each Member

### 6.1 Before starting

For this project, work directly on `main` only when the team has explicitly coordinated that no other member is editing the notebook. Pull the latest changes first:

```bash
git switch main
git pull origin main
```

### 6.2 After completing a section

Save the notebook with `Ctrl + S`, then check:

```bash
git status
```

Stage only the intended files:

```bash
git add notebooks/bank_marketing_ml.ipynb
```

Commit using the message for the section actually completed:

```bash
git commit -m "feat: add baseline classification models"
git push origin main
```

Replace the example message with the appropriate message from Section 5.

If several sections are completed together, one meaningful combined commit is acceptable. Do not create empty commits or split work artificially just to increase the commit count.

### 6.3 If another member has made changes

Before editing, synchronize the notebook. Never overwrite or revert another member's completed sections. Avoid having multiple members edit the same notebook concurrently.

---

## 7. Final Quality Assurance Commit

After Member 5 finishes, one coordinator must validate the complete notebook.

Checklist:

- [ ] Dataset path works on the documented setup.
- [ ] Dataset is excluded from Git.
- [ ] All notebook cells run from top to bottom in a fresh kernel.
- [ ] All required variables are created before use.
- [ ] No stale or contradictory outputs remain.
- [ ] Model metrics match the latest executed outputs.
- [ ] The final model is selected based on cross-validation, not repeated test-set tuning.
- [ ] `duration` is excluded from the prediction features.
- [ ] Documentation and README agree with the notebook.
- [ ] `git status` contains only intended changes.

After fixing any issues, use a commit message such as:

`fix: validate complete notebook execution`

Run:

```bash
git add notebooks/bank_marketing_ml.ipynb docs/ README.md
git commit -m "fix: validate complete notebook execution"
git push origin main
```

Only stage the files that actually changed. If no corrections were needed, there is no need to make a QA commit.

## 8. Optional Future Improvements

After the baseline project is stable, improvements can be added in separate commits.

### Option B — Balanced Improvements

Potential work:

- Classification threshold tuning using validation data.
- Precision-recall curves and average precision.
- Class-weight experiments.
- A more reliable final evaluation strategy.

Suggested commit messages:

- `feat: add classification threshold analysis`
- `feat: add precision recall curve evaluation`
- `feat: compare class weighting strategies`
- `docs: document model improvement findings`

### Option C — Extended Experiments

Potential work:

- Broader hyperparameter searches.
- Additional model candidates.
- More detailed error analysis.
- Additional diagnostics and comparisons.

Suggested commit messages:

- `feat: expand model hyperparameter experiments`
- `feat: add model error analysis`
- `docs: document extended model experiments`

These improvements are optional and should not be mixed into the original baseline history.

---

## 9. Commit Message Convention

Use a consistent prefix:

- `feat:` — new functionality or analysis.
- `docs:` — documentation and explanations.
- `fix:` — corrections and bug fixes.
- `chore:` — repository setup and maintenance.
- `refactor:` — code restructuring without intended behavior changes.

A commit should represent a meaningful, reviewable change. The objective is to preserve a clear history of how the project evolved, not to maximize the number of commits.

## 10. Final Rule

**The notebook is the shared source of truth.** Each member owns their assigned sections, uses the existing variables and preprocessing decisions, and records actual experimental results. The coordinator verifies the integrated notebook before declaring the project complete.
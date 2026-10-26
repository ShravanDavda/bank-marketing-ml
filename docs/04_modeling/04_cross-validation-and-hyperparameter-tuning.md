# Lesson 11 — Cross-Validation and Hyperparameter Tuning

## 1. Objective

Use cross-validation and hyperparameter tuning to investigate whether we can improve the performance of our Bank Marketing classification models.

We will investigate Logistic Regression, Random Forest, and Gradient Boosting. These models are candidates based on the initial evaluation, not final selections.

## 2. Parameters and Hyperparameters

**Model parameters** are learned from training data. For example, Logistic Regression learns coefficients.

**Hyperparameters** are settings chosen before fitting. Examples include regularization strength, tree depth, number of trees, and learning rate.

Choosing suitable hyperparameters can improve generalization, but more complex settings do not automatically produce better results.

## 3. Cross-Validation

Cross-validation repeatedly divides the training data into fitting and validation portions.

In five-fold cross-validation, the data is divided into five folds. Each fold is used for validation once while the remaining folds are used for fitting. The resulting scores are averaged.

We use `StratifiedKFold` because it maintains approximately the same target-class proportions across folds.

## 4. Pipeline and Leakage Prevention

We tune complete scikit-learn pipelines rather than preprocessing the entire dataset before cross-validation.

Each pipeline includes:

- Numerical scaling
- Categorical one-hot encoding
- A classifier

Within every fold, preprocessing is fitted only on that fold's training portion. This prevents validation information from leaking into preprocessing.

## 5. Hyperparameter Search

We use scikit-learn's `GridSearchCV` to evaluate specified combinations of hyperparameters through cross-validation.

The search process:

1. Defines the candidate settings.
2. Fits each candidate across the cross-validation folds.
3. Calculates the selected scoring metric.
4. Identifies the configuration with the highest mean validation score.
5. Refits the best configuration on the full training dataset when `refit=True`.

Searches can be computationally expensive, so we begin with small, reasonable parameter grids.

## 6. Primary and Secondary Metrics

We use F1-score as the primary selection metric for this first tuning pass. F1-score balances precision and recall.

We also record:

- Precision
- Recall
- ROC-AUC

This allows us to understand the trade-offs. A configuration with a higher F1-score may still have lower recall or ROC-AUC.

## 7. Candidate Models

### Logistic Regression

Tune `C`, which controls regularization strength, and optionally `class_weight`.

### Random Forest

Tune the number of trees, maximum tree depth, minimum samples per leaf, and optionally `class_weight`.

### Gradient Boosting

Tune the number of boosting stages, learning rate, and tree depth.

These search spaces are deliberately limited for an initial practical experiment.

## 8. Protecting the Test Set

The held-out test set must not be used for hyperparameter selection.

Cross-validation on the training data guides the search. After selecting a configuration, we will use the held-out test set for the final evaluation.

## 9. Interpreting Search Results

For each model, inspect:

- Best hyperparameters
- Best mean cross-validation F1-score
- Cross-validation precision, recall, and ROC-AUC for the selected configuration
- Computational time and any warnings

Compare the tuned configuration with its baseline using cross-validation results computed on the training data. A higher cross-validation score does not guarantee improved test performance.

## Key Takeaway

Cross-validation gives a more reliable basis for comparing configurations than a single validation split. Hyperparameter tuning searches for better settings, while the held-out test set remains reserved for final evaluation.
# Lesson 10 — Model Comparison

## 1. Objective

Compare the four trained classification models using the evaluation results generated in Lesson 9.

The models are:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The purpose is to understand their strengths, weaknesses, and trade-offs before proceeding to cross-validation and hyperparameter tuning.

## 2. Evaluation Metrics

We compare models using four primary metrics:

- **Precision:** How many predicted subscribers actually subscribed.
- **Recall:** How many actual subscribers the model identified.
- **F1-score:** The harmonic mean of precision and recall.
- **ROC-AUC:** How effectively the model ranks subscribers above non-subscribers across classification thresholds.

Accuracy is also retained for context, but it is not sufficient by itself because the target classes are imbalanced.

## 3. Results from the Initial Test Evaluation

The initial results are:

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6643 | 0.1796 | 0.2827 | 0.7717 |
| Decision Tree | 0.3025 | 0.3280 | 0.3147 | 0.6139 |
| Random Forest | 0.6234 | 0.2316 | 0.3377 | 0.7900 |
| Gradient Boosting | 0.6594 | 0.1994 | 0.3062 | 0.8023 |

Random Forest has the highest F1-score in this initial comparison. Gradient Boosting has the highest ROC-AUC. Decision Tree has the highest recall at the default classification threshold, but its precision and ROC-AUC are lower.

These results describe the current models at their existing configurations and default prediction thresholds.

## 4. Understanding the Trade-offs

### Logistic Regression

Provides a useful baseline and relatively high precision, but its recall is low. It identifies only a limited proportion of actual subscribers at the default threshold.

### Decision Tree

Has the highest recall among the four models at the current threshold. However, its lower precision means many customers predicted as subscribers did not subscribe.

### Random Forest

Has the highest F1-score in the current comparison and a reasonably strong ROC-AUC. It is a promising candidate for further investigation.

### Gradient Boosting

Has the highest ROC-AUC, indicating the strongest ranking performance among these models. Its default-threshold recall remains low, so its operating threshold and other settings may warrant investigation.

## 5. Why We Need Cross-Validation

The test set provides an estimate of performance on unseen data. Repeatedly using test results to choose models or tune hyperparameters can make the final evaluation overly optimistic.

We will use cross-validation on the training data to compare candidate configurations and guide tuning. The held-out test set will be reserved for final evaluation after the modeling decisions have been made.

## 6. Candidate Strategy

For the next stage, we will investigate:

- **Random Forest:** A candidate because of its current F1-score.
- **Gradient Boosting:** A candidate because of its current ROC-AUC.
- **Logistic Regression:** Retained as the baseline.
- **Decision Tree:** Retained as a reference for comparison.

This is a provisional strategy, not a final model selection. Cross-validation results may change the ranking.

## 7. Key Takeaway

No single metric tells the whole story. Model selection must consider precision, recall, F1-score, ranking performance, and the business costs of false positives and false negatives.

The next stage uses cross-validation and hyperparameter tuning on the training data. We will then evaluate the selected final model on the held-out test set.
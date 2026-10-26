# Lesson 9 — Model Evaluation

## 1. Objective

Evaluate the four trained classification models to measure how effectively they predict whether a bank customer subscribes to a term deposit.

The models are evaluated on the held-out test dataset, which was not used to fit them.

## 2. Test Data

- `X_test_model`: Features for the test customers.
- `y_test`: Actual subscription outcomes.
- `pipelines`: The fitted pipelines created in Lesson 8.

The test set must remain separate from training and hyperparameter tuning. Repeatedly using test results to make modeling decisions can produce an overly optimistic estimate of performance.

## 3. Positive Class

We define `yes` as the positive class because identifying subscribers is a key objective of the project.

The `no` class represents customers who did not subscribe.

## 4. Confusion Matrix

A confusion matrix compares actual labels with predicted labels.

- **True Negative (TN):** A non-subscriber correctly predicted as `no`.
- **False Positive (FP):** A non-subscriber incorrectly predicted as `yes`.
- **False Negative (FN):** A subscriber incorrectly predicted as `no`.
- **True Positive (TP):** A subscriber correctly predicted as `yes`.

False positives can waste marketing resources, while false negatives can cause the bank to overlook potential subscribers.

## 5. Evaluation Metrics

### Accuracy
The proportion of all predictions that are correct. It can be misleading when the classes are imbalanced.

### Precision
The proportion of predicted subscribers who actually subscribed. Higher precision means fewer false positives among predicted subscribers.

### Recall
The proportion of actual subscribers identified by the model. Higher recall means fewer subscribers are missed.

### F1-score
The harmonic mean of precision and recall. It provides a single measure that balances both metrics.

### ROC-AUC
Measures how well a model ranks positive examples above negative examples across different classification thresholds. A higher value generally indicates better ranking ability.

ROC-AUC does not measure probability calibration.

## 6. Classification Threshold

The initial predictions from `predict()` generally use the classifier's default decision threshold. For probabilistic classifiers, this is typically 0.5.

Changing the threshold can change precision and recall. Threshold selection should reflect the business objective and should be performed using training or validation data, not repeatedly optimized on the held-out test set.

## 7. Evaluation Strategy

We evaluate all four models using the same test customers and calculate:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

We also inspect confusion matrices and classification reports to understand errors by class.

## 8. Interpretation

Because approximately 11.7% of the dataset's customers subscribed, accuracy alone is insufficient.

We will examine precision, recall, and F1-score closely, while using ROC-AUC to assess ranking ability. The best model cannot be selected from a single metric without considering the intended marketing use case.

## Key Takeaway

Model evaluation measures predictive performance on unseen data. We use multiple complementary metrics to understand the strengths and weaknesses of each model before comparing candidates in the next lesson.
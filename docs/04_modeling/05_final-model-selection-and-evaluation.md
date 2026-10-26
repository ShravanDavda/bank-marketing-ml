# Lesson 12 — Final Model Selection and Evaluation

## 1. Objective

The objective of this lesson is to select a candidate classification model using cross-validation results and evaluate its performance on the held-out test dataset.

In previous lessons, we trained four baseline models, compared their initial performance, and tuned three candidates using five-fold stratified cross-validation.

We will carry the tuned Random Forest forward as our provisional candidate because it achieved the highest mean cross-validation F1-score among the tuned models.

## 2. Review of Cross-Validation Results

The tuned models produced the following results:

| Model | CV F1-score | CV Precision | CV Recall | CV ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.4626 | 0.4675 | 0.4580 | 0.7919 |
| Logistic Regression | 0.3732 | 0.2664 | 0.6228 | 0.7637 |
| Gradient Boosting | 0.3358 | 0.6256 | 0.2297 | 0.7955 |

Random Forest achieved the highest mean cross-validation F1-score. Logistic Regression achieved the highest recall, while Gradient Boosting achieved the highest ROC-AUC.

These differences demonstrate that the models have different strengths and that the selection depends on the intended use case.

## 3. Why Select Random Forest?

We select Random Forest as the provisional candidate because it achieved the highest mean cross-validation F1-score in our current hyperparameter search.

F1-score balances precision and recall. This makes it a reasonable initial selection criterion when we want to identify subscribers while limiting false-positive predictions.

However, this decision is conditional. If the bank prioritizes finding as many potential subscribers as possible, Logistic Regression may deserve further consideration because of its higher cross-validation recall.

The selected model is not guaranteed to be the best model for every business objective.

## 4. Retrieve the Tuned Pipeline

The best fitted pipeline for each candidate is stored in the `tuned_pipelines` dictionary.

The selected Random Forest pipeline contains:

- A fitted numerical scaler.
- A fitted categorical encoder.
- The tuned Random Forest classifier.

The preprocessing steps were fitted within the cross-validation procedure, and the selected pipeline was refitted on the complete training dataset.

We reuse this fitted pipeline for test evaluation instead of fitting a new preprocessing transformer on the test data.

## 5. Generate Test Predictions

We use the selected pipeline to predict outcomes for the customers in `X_test_model`.

The actual outcomes are stored in `y_test`. The predicted class labels are compared with these actual labels to calculate classification metrics.

We also obtain the predicted probability for the `yes` class. These probabilities are used to calculate ROC-AUC.

The test dataset is not used to fit the model.

## 6. Evaluation Metrics

### 6.1 Accuracy

Accuracy measures the proportion of all predictions that are correct.

Accuracy is useful as a general summary but can be misleading for our dataset because approximately 88.3% of the observations belong to the `no` class.

### 6.2 Precision

Precision measures the proportion of predicted subscribers who actually subscribed.

Higher precision means fewer false positives among the customers predicted to subscribe.

### 6.3 Recall

Recall measures the proportion of actual subscribers identified by the model.

Higher recall means fewer actual subscribers are missed.

### 6.4 F1-score

F1-score is the harmonic mean of precision and recall. It provides a single metric for comparing their balance at the current classification threshold.

### 6.5 ROC-AUC

ROC-AUC measures how effectively the model ranks positive examples above negative examples across classification thresholds.

A higher ROC-AUC indicates stronger ranking ability, but it does not guarantee good precision or recall at the default threshold.

## 7. Confusion Matrix

The confusion matrix contains four outcomes:

- **True Negative (TN):** A non-subscriber correctly predicted as `no`.
- **False Positive (FP):** A non-subscriber incorrectly predicted as `yes`.
- **False Negative (FN):** An actual subscriber incorrectly predicted as `no`.
- **True Positive (TP):** An actual subscriber correctly predicted as `yes`.

False positives can lead to marketing resources being spent on customers unlikely to subscribe. False negatives can cause the bank to overlook potential subscribers.

The relative importance of these errors depends on the bank's costs and marketing objectives.

## 8. Compare the Tuned Model with the Baseline

We compare the tuned Random Forest with the original baseline Random Forest using the same held-out test examples.

We inspect accuracy, precision, recall, F1-score, and ROC-AUC.

A higher test score after tuning is encouraging, but the comparison should not be interpreted as an independent estimate of the tuning improvement because the test results were already inspected during earlier model comparison.

## 9. Test-Set Reuse Limitation

The held-out test set was evaluated in Lesson 9, and those results influenced our candidate selection in subsequent lessons.

Therefore, it is not a completely untouched final benchmark.

We can still report the tuned model's performance on this test set, but we must acknowledge that repeated inspection may make our final results less independent than intended.

For a stronger final benchmark, we would need a fresh, independent dataset or holdout set that has not influenced model selection.

## 10. Classification Threshold

The predictions generated by `predict()` use the classifier's default decision rule. For standard probabilistic binary classifiers, this is generally equivalent to choosing the class with the highest probability, which often corresponds to a 0.5 threshold for the positive class.

Changing the threshold can alter precision and recall without retraining the model.

Threshold selection should be guided by validation data and the bank's business objectives, not repeatedly optimized on the held-out test set.

## 11. Interpretation of Results

After running the notebook, we record:

- The final test accuracy.
- The final test precision, recall, F1-score, and ROC-AUC.
- The numbers of true positives, false positives, true negatives, and false negatives.
- Whether the tuned model's test metrics differ from the baseline.
- Whether its precision and recall are suitable for the intended use case.

We must use the actual notebook outputs rather than assume tuning improved performance.

## 12. Key Takeaway

Random Forest is the provisional candidate because it achieved the highest mean cross-validation F1-score in our current search.

We evaluate this candidate on the held-out test dataset, inspect its confusion matrix, and compare it with the baseline. We also document the test-set reuse limitation so that the project report does not overstate the independence of the final evaluation.
# Lesson 14 — Conclusion and Future Work

## 1. Project Overview

The Bank Marketing ML project investigates whether machine learning can predict if a bank customer will subscribe to a term deposit.

The project follows a practical classification workflow, covering dataset understanding, data cleaning, exploratory data analysis, feature engineering, preprocessing, model training, evaluation, hyperparameter tuning, and model interpretation.

The objective is to develop a predictive model and understand its performance, strengths, limitations, and potential use in marketing prioritization.

## 2. Problem and Dataset

The primary dataset, `bank-full.csv`, contains 45,211 records and 17 original columns: 16 input features and one target column, `y`.

The target is binary:

- `yes`: The customer subscribed to a term deposit.
- `no`: The customer did not subscribe.

Approximately 11.7% of the records belong to the `yes` class, while 88.3% belong to the `no` class. This class imbalance makes accuracy alone an insufficient measure of model performance.

The intended use case is to predict before contacting a customer. Therefore, the `duration` feature was excluded from model inputs because its value is unavailable at that prediction time.

## 3. Approach

The project followed these stages:

1. Defined the prediction problem and examined the dataset.
2. Audited data quality and explored feature distributions and relationships.
3. Created the `previously_contacted` feature from `pdays`.
4. Split the data into stratified training and test subsets.
5. Scaled numerical features and one-hot encoded categorical features.
6. Trained four baseline classifiers.
7. Evaluated and compared baseline performance.
8. Tuned three candidate models using five-fold stratified cross-validation.
9. Evaluated the selected tuned Random Forest on the held-out test dataset.
10. Investigated model feature importance using two methods.

The preprocessing operations were included in model pipelines so that they could be fitted correctly during cross-validation.

## 4. Models Investigated

Four baseline classifiers were trained:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Logistic Regression was retained as a baseline. Random Forest and Gradient Boosting were promising candidates based on different evaluation metrics.

Hyperparameter tuning was performed for Logistic Regression, Random Forest, and Gradient Boosting. Random Forest was carried forward because it achieved the highest mean cross-validation F1-score among the tuned candidates.

## 5. Final Model Results

The tuned Random Forest produced the following test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 0.8796 |
| Precision | 0.4850 |
| Recall | 0.4745 |
| F1-score | 0.4797 |
| ROC-AUC | 0.8010 |

The confusion matrix contained:

- True negatives: 7,452
- False positives: 533
- False negatives: 556
- True positives: 502

The model identified 502 of the 1,058 actual subscribers in the test dataset.

## 6. Baseline and Tuned Model Comparison

The baseline Random Forest and tuned Random Forest produced the following test results:

| Metric | Baseline | Tuned |
|---|---:|---:|
| Accuracy | 0.8937 | 0.8796 |
| Precision | 0.6234 | 0.4850 |
| Recall | 0.2316 | 0.4745 |
| F1-score | 0.3377 | 0.4797 |
| ROC-AUC | 0.7900 | 0.8010 |

The tuned model achieved higher recall, F1-score, and ROC-AUC. However, its precision and accuracy were lower.

This demonstrates a precision-recall trade-off: the tuned model identifies more actual subscribers but also produces more false-positive predictions.

Whether this trade-off is acceptable depends on the bank's marketing objectives and the cost of contacting customers who do not subscribe.

## 7. Model Interpretation Findings

Two feature-importance methods were applied to the selected Random Forest.

### Built-in Feature Importance

The highest-ranked original features included:

- `balance`
- `month`
- `age`
- `day`
- `job`
- `poutcome`

This method summarizes how features contributed to impurity reduction across the fitted trees.

### Permutation Importance

The highest mean decreases in F1-score after shuffling features were observed for:

- `month`
- `contact`
- `housing`
- `poutcome`
- `day`

The two methods produced different rankings because they measure different aspects of model behavior.

These findings describe predictive patterns and do not establish that the features cause a customer to subscribe. Correlated features and the evaluation dataset can also affect the results.

## 8. Limitations

### 8.1 Class Imbalance

Subscribers represent a minority of the dataset. Accuracy alone would not adequately describe the model's ability to identify subscribers.

### 8.2 Precision and Recall Trade-off

The tuned Random Forest identifies more subscribers than the baseline, but its precision is lower. The preferred balance depends on the business objective.

### 8.3 Test-Set Reuse

The test set was inspected during earlier model comparison and influenced subsequent modeling decisions. It is therefore not a completely untouched final benchmark.

The reported results should be interpreted with this limitation in mind.

### 8.4 Feature-Importance Limitations

Feature importance does not prove causation. Correlated features and the selected importance method can affect the rankings.

### 8.5 Operational Validation

The project evaluates predictive performance but does not establish financial return, the optimal marketing threshold, or the effect of deploying the model in a real campaign.

## 9. Future Work

Potential future improvements include:

1. Evaluate the selected model on a fresh, untouched holdout dataset.
2. Select a classification threshold using validation data and a defined business objective.
3. Compare alternative thresholds using precision, recall, F1-score, and ROC-AUC where appropriate.
4. Evaluate probability calibration if the predicted probabilities will guide marketing decisions.
5. Investigate performance across customer groups and campaign periods.
6. Examine the impact of correlated features on importance estimates.
7. Build a reproducible prediction interface and validate the model before operational deployment.
8. Monitor performance if the model is deployed and retrain only when justified by evidence.

These are proposed improvements, not completed project features.

## 10. Conclusion

This project developed a practical machine learning workflow for predicting bank term-deposit subscriptions using historical marketing data.

Four baseline classifiers were evaluated, and three candidates were tuned using five-fold stratified cross-validation. Random Forest achieved the highest mean cross-validation F1-score among the tuned candidates and was carried forward for test evaluation.

The tuned Random Forest achieved a test F1-score of 0.4797 and ROC-AUC of 0.8010. Compared with the baseline Random Forest, it identified more actual subscribers but also produced more false positives.

Feature-importance analysis highlighted customer balance, campaign timing, age, contact method, and previous campaign outcome as influential predictive features, depending on the method used.

The project demonstrates how machine learning can identify patterns in historical marketing data. However, model performance must be considered alongside class imbalance, precision-recall trade-offs, feature-importance limitations, and the need for independent evaluation before operational use.
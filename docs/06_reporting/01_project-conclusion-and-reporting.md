# Lesson 14 — Project Conclusion and Reporting

## 1. Introduction

The Bank Marketing ML project investigates whether machine learning can predict if a bank customer will subscribe to a term deposit.

The project follows a practical classification workflow, including data understanding, cleaning, exploratory data analysis, feature engineering, preprocessing, model training, evaluation, hyperparameter tuning, and model interpretation.

The objective is to build and evaluate a predictive model while understanding its performance, limitations, and potential use in marketing prioritization.

## 2. Problem Statement

Banks conduct marketing campaigns to encourage customers to subscribe to financial products. Identifying customers who are more likely to subscribe may help prioritize marketing resources.

This project uses historical bank marketing data to predict a binary target:

- `yes`: The customer subscribed to a term deposit.
- `no`: The customer did not subscribe.

The intended prediction is made before contacting the customer. Therefore, the `duration` feature is excluded because its value is not available at that prediction time.

The model estimates subscription likelihood from historical patterns. It does not guarantee that a customer will subscribe.

## 3. Dataset Overview

The primary dataset, `bank-full.csv`, contains 45,211 records and 17 original columns: 16 input features and one target.

The dataset includes numerical and categorical variables describing customer characteristics, account information, contact details, and campaign history.

The target is imbalanced:

- Approximately 88.3% of customers belong to the `no` class.
- Approximately 11.7% belong to the `yes` class.

This imbalance makes it important to evaluate precision, recall, F1-score, and ROC-AUC rather than relying on accuracy alone.

## 4. Data Preparation

The data preparation workflow included:

1. Inspecting dataset dimensions, data types, and target distribution.
2. Auditing missing values, duplicate records, categorical values, and numerical ranges.
3. Performing exploratory data analysis to investigate feature distributions and relationships with subscription.
4. Creating `previously_contacted` from the `pdays` variable while retaining the original feature.
5. Splitting the data into stratified training and test subsets.
6. Scaling numerical features and one-hot encoding categorical features.

The preprocessing steps were organized into a scikit-learn pipeline. This allows preprocessing to be fitted within each cross-validation training fold and applied consistently to validation data.

The `duration` feature was excluded from model inputs to preserve the pre-contact prediction use case.

## 5. Baseline Models

Four classification models were trained and initially evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The baseline results demonstrated that different models performed differently across precision, recall, F1-score, and ROC-AUC.

Random Forest achieved the highest initial F1-score of 0.3377, while Gradient Boosting achieved the highest initial ROC-AUC of 0.8023.

These results motivated further investigation of the ensemble models while retaining Logistic Regression as a baseline.

## 6. Cross-Validation and Hyperparameter Tuning

Logistic Regression, Random Forest, and Gradient Boosting were tuned using five-fold stratified cross-validation.

F1-score was used as the primary selection metric, while precision, recall, and ROC-AUC were recorded as secondary metrics.

The best mean cross-validation F1-scores were:

| Model | CV F1-score | CV Precision | CV Recall | CV ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.4626 | 0.4675 | 0.4580 | 0.7919 |
| Logistic Regression | 0.3732 | 0.2664 | 0.6228 | 0.7637 |
| Gradient Boosting | 0.3358 | 0.6256 | 0.2297 | 0.7955 |

Random Forest achieved the highest mean cross-validation F1-score among the tuned candidates and was carried forward as the provisional final model.

Logistic Regression had higher cross-validation recall, illustrating the trade-off between identifying more subscribers and limiting false positives.

## 7. Final Model Evaluation

The tuned Random Forest was evaluated on the held-out test dataset.

The recorded results were:

| Metric | Tuned Random Forest |
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

The model identified 502 of the 1,058 actual subscribers in the test set.

Compared with the baseline Random Forest, the tuned model achieved higher recall, F1-score, and ROC-AUC, but lower accuracy and precision.

The tuned model therefore identifies more potential subscribers at the cost of more false-positive predictions.

## 8. Model Interpretation

Two feature-importance methods were used to investigate the selected Random Forest.

### 8.1 Built-in Feature Importance

The leading original features by aggregated built-in importance included:

- `balance`
- `month`
- `age`
- `day`
- `job`
- `poutcome`

These features contributed to impurity reduction across the fitted trees.

### 8.2 Permutation Importance

Permutation importance measured the decrease in F1-score when each original feature was shuffled.

The highest mean decreases were observed for:

- `month`: approximately 0.1154
- `contact`: approximately 0.1000
- `housing`: approximately 0.0425
- `poutcome`: approximately 0.0354

The methods produced different rankings because they measure different aspects of model behavior.

These results describe predictive relationships. They do not establish causal effects, and correlated features can affect individual importance estimates.

## 9. Limitations

### 9.1 Class Imbalance

The target contains substantially more non-subscribers than subscribers. Accuracy alone would therefore provide an incomplete picture of model performance.

### 9.2 Precision and Recall Trade-off

The tuned Random Forest improves recall compared with the baseline but produces lower precision. Whether this trade-off is desirable depends on the bank's marketing costs and objectives.

### 9.3 Test-Set Reuse

The test set was inspected during earlier model comparison and influenced subsequent modeling decisions. Therefore, the reported test performance is not an entirely independent final benchmark.

A future evaluation should use a fresh, untouched holdout dataset or an appropriate later-period dataset.

### 9.4 Prediction-Time Constraints

The `duration` feature was excluded because it is unavailable before contacting a customer. This restriction should remain in place for any operational version of the model.

### 9.5 Feature-Importance Limitations

Feature importance does not demonstrate causation. Results can be influenced by correlated features, the model architecture, and the selected scoring metric.

### 9.6 Operational and Business Validation

The current project evaluates predictive performance. It does not establish the financial return of using the model, the optimal marketing threshold, or whether deployment would improve campaign outcomes.

## 10. Future Improvements

Potential improvements include:

1. Evaluate the selected model on an untouched holdout dataset.
2. Select the operating threshold using validation data and a defined business objective.
3. Compare performance using alternative threshold-independent and threshold-dependent metrics.
4. Investigate the effects of correlated features and categorical encoding on feature importance.
5. Validate performance across customer groups and campaign periods.
6. Assess model calibration if predicted probabilities will be used for decision-making.
7. Establish a reproducible prediction workflow and monitor performance if the model is deployed.

These are potential next steps rather than completed project features.

## 11. Conclusion

This project developed a binary classification workflow for predicting bank term-deposit subscriptions from historical marketing data.

Four baseline models were evaluated, and three candidates were tuned using five-fold stratified cross-validation. Random Forest achieved the highest mean cross-validation F1-score and was selected for further evaluation.

The tuned Random Forest achieved a test F1-score of 0.4797 and ROC-AUC of 0.8010. Its recall increased relative to the baseline, while precision and accuracy decreased.

Feature-importance analysis identified customer balance, campaign timing, age, contact method, and previous campaign outcome as influential predictive features, depending on the importance method used.

Overall, the results demonstrate that machine learning can identify useful patterns in historical bank marketing data. However, model selection, operating thresholds, data limitations, and business costs must be considered before operational use. The test-set reuse limitation should also be acknowledged when reporting the results.
# Lesson 13 — Model Interpretation

## 1. Objective

The objective of this lesson is to understand which input features contribute most to the predictions of our selected Random Forest model.

We will use two methods:

1. Built-in Random Forest feature importance.
2. Permutation importance.

These methods provide complementary views of feature importance and help us communicate the model's behavior in the final project report.

## 2. Why Model Interpretation Matters

Predictive performance tells us how well a model predicts outcomes, but it does not fully explain which patterns the model uses.

Model interpretation helps us:

- Understand which features the model relies on.
- Communicate the model's behavior.
- Identify potentially misleading or unexpected patterns.
- Discuss limitations and possible sources of bias.

Feature importance describes relationships used by a predictive model. It does not establish that a feature causes a customer to subscribe.

## 3. Built-in Random Forest Feature Importance

Random Forest consists of multiple decision trees. Each tree splits observations using features to reduce impurity.

The classifier exposes these importance values through `feature_importances_`. For a Random Forest, the values are aggregated across its trees and normalized.

Higher importance indicates greater contribution to impurity reduction within the fitted forest.

### Limitations

- Impurity-based importance can favor features with many possible split points.
- Correlated features can divide importance between themselves.
- Importance does not show whether a feature increases or decreases the probability of subscription.
- Importance values do not directly measure causal effects.

## 4. Feature Names After Preprocessing

Our model pipeline uses a `ColumnTransformer` to scale numerical features and one-hot encode categorical features.

Consequently, the transformed feature matrix may contain more columns than the original input dataset. A categorical feature such as `job` can become multiple encoded columns.

We must obtain feature names from the fitted preprocessing pipeline so that importance values are matched to the correct transformed columns.

When interpreting categorical features, it can be useful to aggregate the importance of their encoded columns to estimate their combined contribution.

## 5. Permutation Importance

Permutation importance measures how much a model's evaluation score changes when a feature is randomly shuffled.

The procedure is:

1. Calculate the model's baseline score.
2. Shuffle one feature's values.
3. Recalculate the score.
4. Measure the decrease in performance.
5. Repeat the shuffle and average the results.

A larger positive decrease suggests that the model relies more on the feature for the selected metric.

### Why use permutation importance?

It evaluates the effect of disrupting a feature on the model's predictive score rather than relying only on tree-splitting statistics.

### Limitations

- Correlated features can substitute for one another.
- Results depend on the chosen scoring metric and evaluation dataset.
- Shuffling creates feature combinations that may be unrealistic.
- Small differences in importance may not be meaningful.

## 6. Selected Scoring Metric

We use F1-score as the primary permutation-importance scoring metric because it balances precision and recall.

This choice aligns with the metric used to select our current Random Forest candidate.

Different metrics can produce different importance rankings. If the business objective changes, we may need to interpret importance using another metric, such as ROC-AUC or recall.

## 7. Evaluation Dataset and Limitations

Permutation importance can be calculated on training, validation, or test data, depending on the question being investigated.

For this project, we will use the held-out test dataset to describe how the selected model relies on features when evaluated on these observations.

However, we already inspected this test set during model comparison and final evaluation. Therefore, it is not an untouched benchmark. Our interpretation should acknowledge this limitation and should not use the importance results to select or tune a new model.

## 8. Interpreting the Results

When inspecting the feature-importance plots, we should ask:

- Which transformed features have the highest built-in importance?
- Which original categorical variables contribute across multiple encoded columns?
- Which features produce the largest decrease in F1-score when shuffled?
- Do the two importance methods identify similar influential features?
- Are any results potentially affected by correlated features or data limitations?

Agreement between the methods can provide useful supporting evidence, but it does not prove that the identified features cause subscription behavior.

## 9. Important Prediction-Time Constraint

The intended use case is to predict subscription before contacting a customer. The `duration` feature is unavailable at that point and remains excluded from model inputs.

This constraint must also be respected when interpreting the model. We should not use information unavailable at prediction time to explain or improve the operational prediction workflow.

## 10. Key Takeaway

Built-in Random Forest importance and permutation importance provide complementary views of the features used by our model.

We will use them to describe predictive patterns, not causal relationships. The results will support the project report by explaining which features appear most influential and by documenting the limitations of those conclusions.
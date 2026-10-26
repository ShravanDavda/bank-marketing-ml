# Lesson 8 — Model Training

## 1. Objective

Train four classification models to predict whether a bank customer will subscribe to a term deposit.

The target column is `y`, with values `yes` and `no`.

## 2. What Is Model Training?

Model training is the process of learning patterns from historical examples with known outcomes.

- **Features (`X`):** Information used to make predictions.
- **Target (`y`):** The outcome we want to predict.
- **Training data:** Examples used to fit the model.
- **Fitted model:** A trained algorithm that can generate predictions.

In scikit-learn, `model.fit(X_train, y_train)` fits the model using the training examples and their labels.

## 3. The Four Models

### 3.1 Logistic Regression

A classification algorithm that estimates the probability of a binary outcome using a weighted combination of input features. It is our baseline model and is relatively easy to interpret.

### 3.2 Decision Tree Classifier

Learns feature-based decision rules. It can capture nonlinear relationships but may overfit when the tree becomes too complex.

### 3.3 Random Forest Classifier

Combines multiple decision trees trained with randomness in sampling and feature selection. Combining trees can make predictions more robust than relying on a single tree.

### 3.4 Gradient Boosting Classifier

Builds decision trees sequentially. Later trees improve the ensemble by addressing patterns that the existing model has not captured adequately.

We will compare all four models rather than assume one is best.

## 4. Why Use a Pipeline?

A scikit-learn `Pipeline` combines preprocessing and classification into one workflow.

Our preprocessing consists of:

- `StandardScaler`: Scales numerical features.
- `OneHotEncoder`: Converts categorical features into numerical indicator columns.
- `ColumnTransformer`: Applies the appropriate transformations to each feature group.

We use `handle_unknown="ignore"` so the encoder can process categories not encountered during fitting.

A pipeline also helps prevent data leakage during cross-validation because preprocessing can be fitted separately within each training fold.

## 5. Training Data and Test Data

We train using `X_train_model` and `y_train`.

We reserve `X_test_model` and `y_test` for final held-out evaluation. The test set should not guide routine model selection or hyperparameter tuning.

## 6. Excluding Duration

Our intended use case is to prioritize customers before contacting them. The `duration` feature is unavailable at that point, so it is excluded from model inputs.

The original DataFrame remains unchanged. This is a deliberate prediction-time constraint.

## 7. Training Time

We record elapsed training time for each model. Training time helps us understand computational cost, but it does not measure predictive quality.

The fastest model is not necessarily the best model.

## 8. Training Versus Evaluation

Training fits an algorithm to historical data. Evaluation measures its predictive performance on data it has not learned from.

In the next lesson, we will compare models using confusion matrices, precision, recall, F1-score, and ROC-AUC. Because the target is imbalanced, accuracy alone is insufficient.

## 9. Reproducibility

We use fixed random seeds where supported so that results are easier to reproduce under the same software and environment.

## Key Takeaway

Train all four models using the same leakage-safe preprocessing workflow. Evaluate their performance separately and select a model based on evidence rather than assumptions.
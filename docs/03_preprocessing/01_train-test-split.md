# Train-Test Split — Bank Marketing ML

## 1. What Is Train-Test Split?

Train-test split is the process of dividing a dataset into separate subsets for model training and evaluation.

The training set is used to learn patterns from historical data. The test set is held aside to evaluate how well the trained model performs on records it has not seen during training.

The purpose is to estimate how well a model generalizes to unseen data.

## 2. Why Do We Need It?

A model may perform well on the records it learned from but perform poorly on new records. This can happen when it memorizes patterns specific to the training data.

Evaluating on a separate test set helps us assess generalization.

The workflow is:

```text id="x41y7n"
Complete Dataset
       |
       v
Train-Test Split
       |
       +------> Training Set
       |             |
       |             v
       |        Model Training
       |
       +------> Test Set
                     |
                     v
              Final Evaluation
```

The test set should remain separate from model fitting and preprocessing decisions.

## 3. Our Dataset

Our primary dataset is `bank-full.csv`.

After creating `previously_contacted`, assuming no other columns have been added or removed, the DataFrame contains:

- 45,211 records
- 17 original columns
- 1 additional engineered feature
- 18 columns in total, including the target

The target is `y`, which contains `yes` and `no`.

We need to separate the input features from the target before splitting.

## 4. Features (`X`) and Target (`y`)

### Features

`X` contains the input variables used by the model.

```python id="v41exj"
X = df.drop(columns=["y"])
```

This removes the target column from the input data.

### Target

`y` contains the outcome we want to predict.

```python id="xf12jx"
y = df["y"]
```

The model learns from the training features and their corresponding target labels.

Conceptually:

```text id="q91m1j"
X_train + y_train
         |
         v
     Train Model
         |
         v
  Predict X_test
         |
         v
 Compare with y_test
```

## 5. The 80:20 Split

A common starting point is to use 80% of the records for training and 20% for testing.

| Subset | Proportion | Purpose |
|---|---:|---|
| Training set | 80% | Fit the model |
| Test set | 20% | Evaluate the trained model |

For 45,211 records, this produces approximately:

- Training records: 36,168–36,169
- Test records: 9,042–9,043

The exact counts depend on the split function's rounding behavior.

An 80:20 split is a reasonable starting point, not a universal rule. Other proportions may be suitable for other datasets and project requirements.

## 6. Using `train_test_split()`

Scikit-learn provides the `train_test_split()` function.

```python id="s4jfj1"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Important parameters

**`test_size=0.2`**

Assigns 20% of the records to the test set. The remaining records are used for training.

**`random_state=42`**

Sets a random seed so that the split can be reproduced when the dataset and code remain unchanged.

The number 42 is an arbitrary choice; it does not make the model inherently better.

**`stratify=y`**

Attempts to preserve the target-class proportions in both subsets.

This is useful for our project because `y` is imbalanced.

## 7. Understanding Stratification

Our dataset contains approximately:

- 88.3% `no`
- 11.7% `yes`

If we split the data randomly without stratification, the proportions in each subset may differ from the original distribution.

Using `stratify=y` helps maintain similar proportions of both classes in the training and test sets.

This is especially helpful when evaluating a classifier on an imbalanced target.

Stratification does not balance the classes. It preserves approximately the same imbalance in both subsets.

## 8. Understanding the Four Outputs

The split function returns four objects.

| Variable | Meaning |
|---|---|
| `X_train` | Training input features |
| `X_test` | Test input features |
| `y_train` | Training target labels |
| `y_test` | Test target labels |

The training data is used to fit the model. The test features are used to generate predictions, which are compared with the test labels.

The test set should not be used to fit the model.

## 9. Verify the Split

We should inspect the dimensions after splitting.

```python id="wqj5ab"
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
```

We can also check the target distribution:

```python id="o7ngf4"
print((y_train.value_counts(normalize=True) * 100).round(2))
print((y_test.value_counts(normalize=True) * 100).round(2))
```

The training and test sets should have approximately the same `yes` and `no` percentages because stratification was used.

### Check for overlap

```python id="y7p6sm"
assert len(X_train) + len(X_test) == len(X)
assert X_train.index.intersection(X_test.index).empty
assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)
```

These assertions verify that:

- All records are accounted for.
- Training and test indices do not overlap.
- Features and target labels remain aligned within each subset.

## 10. Data Leakage Prevention

Data leakage occurs when information that should not be available to the model influences training or evaluation.

One common source of leakage is fitting preprocessing transformations on the complete dataset before splitting it.

For example, a scaler calculates statistics such as the mean and standard deviation. If it learns those statistics from the test set, information from the held-out data has influenced the training process.

The correct sequence is:

```text id="z5nvcc"
Split Dataset
      |
      v
Fit Preprocessing on Training Data
      |
      v
Transform Training Data
      |
      v
Transform Test Data Using the Fitted Preprocessor
```

We will implement this properly in Lesson 7 using suitable preprocessing techniques and, where appropriate, a Scikit-learn pipeline.

## 11. Test Set vs Validation Set

These are different concepts.

- **Training set:** Used to fit the model.
- **Validation set:** Used to compare candidate models and tune their settings.
- **Test set:** Used for final evaluation on held-out data.

A separate validation set is not always necessary. Cross-validation on the training data can be used for model comparison and hyperparameter tuning.

For our project, we can reserve the test set and use cross-validation on the training data during model selection.

This helps avoid repeatedly using the test set to make development decisions.

## 12. Prediction-Time Feature Availability

The `duration` feature records the duration of the last contact in seconds.

If our intended use case is predicting subscription before contacting a customer, duration is not available at prediction time. Using it would make the model unsuitable for that use case.

We must settle the prediction-time definition before finalizing the model's input features.

A correct train-test split cannot fix leakage caused by an inappropriate feature.

## 13. Common Mistakes

1. Training and evaluating a model on the same records.
2. Forgetting to separate the target from the input features.
3. Using an inconsistent random seed when reproducibility is required.
4. Ignoring class imbalance during splitting.
5. Fitting scalers or encoders on the full dataset before splitting.
6. Using the test set repeatedly to choose models and hyperparameters.
7. Assuming stratification balances the target classes.
8. Ignoring whether features will be available at prediction time.
9. Forgetting to verify that features and labels remain aligned.

## 14. Connection to Our Notebook

This material corresponds to:

`notebooks/bank_marketing_ml.ipynb`

Section:

`8. Train-Test Split`

The notebook will contain the split implementation, dimensions, target proportions, and verification checks. This document provides the fuller explanation for learning and revision.

## 15. Quick Revision

**What is train-test split?**  
Dividing data into separate training and test subsets.

**Why do we need it?**  
To evaluate model performance on records not used for fitting.

**What does `test_size=0.2` mean?**  
20% of the records are assigned to testing.

**What does `random_state=42` do?**  
Makes the randomized split reproducible.

**What does `stratify=y` do?**  
Preserves approximately the same target-class proportions in both subsets.

**What are `X` and `y`?**  
`X` contains the input features; `y` contains the target labels.

**Should we fit a scaler before splitting?**  
No. Fit preprocessing transformations on training data only.

**What is the role of the test set?**  
To evaluate the selected model on held-out records.

## 16. Key Takeaway

Train-test split separates model learning from final evaluation. For our project, we use an 80:20 split with a fixed random seed and target stratification, then verify the split before proceeding to preprocessing and model development.
# Data Preprocessing — Bank Marketing ML

## 1. What Is Data Preprocessing?

Data preprocessing is the process of transforming input data into a format suitable for machine learning algorithms.

Our Bank Marketing dataset contains numerical and categorical features. Some models require categorical values to be converted into numerical representations, while some benefit from numerical features being scaled.

Our objective is to build a reusable preprocessing pipeline that prepares both training and test data consistently.

## 2. Why Do We Need Preprocessing?

Consider these features from our dataset:

| Feature | Example | Issue |
|---|---|---|
| `age` | 42 | Numerical feature that may benefit from scaling |
| `balance` | 1500 | Numerical feature with a different range |
| `job` | technician | Categorical text |
| `marital` | married | Categorical text |
| `previously_contacted` | 1 | Already represented numerically |

Many machine learning algorithms cannot directly process raw category strings. Scaling can also help algorithms that are sensitive to the magnitudes of input features.

Preprocessing ensures that the features are represented appropriately for the algorithms we intend to use.

## 3. Feature Engineering vs Preprocessing

Feature engineering creates or transforms useful features. Preprocessing prepares the features for a machine learning algorithm.

| Feature engineering | Preprocessing |
|---|---|
| Create `previously_contacted` from `pdays` | Encode `job` using one-hot encoding |
| Derive a meaningful indicator | Scale numerical features |
| Incorporate domain knowledge | Apply consistent transformations |

Our `previously_contacted` feature was created in Lesson 5. We will now preprocess it alongside the other input features.

## 4. One-Hot Encoding

### 4.1 What is one-hot encoding?

One-hot encoding represents each category using a separate binary column.

For example, imagine the `marital` feature contains three categories:

| Original category | `marital_divorced` | `marital_married` | `marital_single` |
|---|---:|---:|---:|
| divorced | 1 | 0 | 0 |
| married | 0 | 1 | 0 |
| single | 0 | 0 | 1 |

The encoder creates numerical representations without assigning an artificial ranking to the categories.

### 4.2 Why use it in our project?

Our dataset contains categorical features such as:

- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`
- `contact`
- `month`
- `poutcome`

We will use Scikit-learn's `OneHotEncoder`.

```python id="8mks9k"
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
```

The parameter `handle_unknown="ignore"` allows transformation to continue when a category appears that was not observed during fitting. For that feature, the unseen category is represented by zeros across the encoder's known category columns.

This is useful when applying the fitted encoder to unseen records.

## 5. Numerical Scaling

### 5.1 Why scale numerical features?

Numerical features can have very different ranges.

For example:

- `age` may be in the tens.
- `balance` may reach much larger values.
- `duration` is measured in seconds.

For algorithms sensitive to feature magnitudes, large numerical ranges can affect optimization and model behavior.

Tree-based models such as Decision Trees and Random Forests generally do not require standardization, but Logistic Regression commonly benefits from it.

### 5.2 StandardScaler

StandardScaler transforms a feature using its mean and standard deviation.

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) is the original value.
- \(\mu\) is the mean learned from the training data.
- \(\sigma\) is the standard deviation learned from the training data.
- \(z\) is the standardized value.

```python id="98x5kj"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

After fitting, the scaler applies the learned parameters to other data.

Standardization does not guarantee that the transformed data is normally distributed. It changes the location and scale of the feature.

## 6. ColumnTransformer

Our dataset contains numerical and categorical features that require different preprocessing operations.

Scikit-learn's `ColumnTransformer` allows us to apply separate transformations to selected columns.

For our project:

- Numerical features: use `StandardScaler`.
- Categorical features: use `OneHotEncoder`.

This keeps the preprocessing logic organized and ensures that the appropriate transformation is applied to each feature group.

## 7. Pipeline

A `Pipeline` combines preprocessing and a machine learning estimator into a single workflow.

For example, a later model pipeline may contain:

```text id="b4mt8c"
Input Features
      ↓
ColumnTransformer
  ├── Scale numerical features
  └── Encode categorical features
      ↓
Classification Model
      ↓
Prediction
```

This structure is useful because it allows preprocessing to be fitted correctly inside cross-validation and hyperparameter tuning.

We will use the fitted preprocessing object during this lesson and combine it with classifiers during model training.

## 8. Preventing Data Leakage

Data leakage occurs when information from data that should remain unseen influences model fitting or evaluation.

A common preprocessing mistake is fitting a scaler or encoder on the entire dataset before splitting it.

Incorrect:

```python id="u7m0dc"
# Avoid fitting on all records before splitting
X_processed = preprocessor.fit_transform(X)
```

Correct:

```python id="w7h8tz"
preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

The encoder learns the available categories from training data, and the scaler learns its numerical statistics from training data.

The fitted transformations are then applied to the test set without learning from it.

**Important:** When using cross-validation, the safest approach is to put preprocessing and the classifier in one Scikit-learn `Pipeline`. This lets each cross-validation fold fit its transformations using that fold's training portion.

## 9. Prediction-Time Feature Availability

Our intended use case is to predict subscription before contacting a customer.

The `duration` feature represents the duration of the last contact in seconds. Before a new contact occurs, that duration is not available.

Therefore, we will exclude `duration` from the model's input features for this use case.

This is a prediction-time availability decision, not a scaling or encoding decision.

We retain the original column in `df` so the source dataset remains intact.

## 10. Our Preprocessing Strategy

Our project uses the following decisions:

| Feature group | Treatment |
|---|---|
| Numerical features | StandardScaler |
| Categorical features | OneHotEncoder |
| `previously_contacted` | Treat as a numerical binary feature |
| Target `y` | Keep separate from input features |
| `duration` | Exclude from pre-contact model inputs |

We will not scale or encode the target as part of the feature preprocessor. The target remains the original `yes` and `no` labels for classification.

## 11. Verifying the Result

After preprocessing, inspect the resulting shapes.

```python id="xg6fr5"
print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original test shape:", X_test.shape)
print("Processed test shape:", X_test_processed.shape)
```

The number of processed columns will generally be greater than the number of original input columns because one-hot encoding expands categorical features into multiple columns.

We should also verify that the output contains no non-finite numerical values.

For sparse output, use an appropriate sparse-matrix check rather than converting a large matrix to a dense array unnecessarily.

## 12. Common Mistakes

1. **Passing categorical strings directly to an algorithm that requires numerical input.**  
   Encode categorical variables appropriately.

2. **Scaling before the train-test split.**  
   Fit scaling transformations on training data only.

3. **Fitting an encoder on the complete dataset.**  
   This allows information about test categories to influence preprocessing.

4. **Assuming every model needs scaling.**  
   Scaling requirements depend on the algorithm.

5. **Treating categories as ordered numbers without justification.**  
   Arbitrary category numbers can imply a false ordering.

6. **Transforming the target as if it were an input feature.**  
   Keep `y` separate from the feature preprocessor.

7. **Ignoring unseen categories.**  
   Configure the encoder to handle categories not observed during fitting.

8. **Using features unavailable at prediction time.**  
   Our pre-contact prediction scenario excludes `duration`.

9. **Preprocessing outside cross-validation.**  
   Fit preprocessing within each training fold when evaluating models with cross-validation.

10. **Converting large sparse matrices to dense arrays unnecessarily.**  
    One-hot encoding can create many columns, so sparse representations are often more memory-efficient.

## 13. Connection to Our Notebook

This document corresponds to:

`notebooks/bank_marketing_ml.ipynb`

Section:

`9. Data Preprocessing`

Planned subsections:

- 9.1 Define Model Input Features
- 9.2 Identify Numerical and Categorical Features
- 9.3 Build the Preprocessor
- 9.4 Fit on Training Data
- 9.5 Transform Training and Test Data
- 9.6 Verify the Output

The notebook contains the implementation, actual output shapes, and concise interpretations. This document provides the fuller explanations for learning and revision.

## 14. Quick Revision

**What is preprocessing?**  
Preparing features in a suitable format for machine learning algorithms.

**What does one-hot encoding do?**  
Converts categorical values into binary indicator columns.

**What does StandardScaler do?**  
Standardizes numerical features using the mean and standard deviation learned during fitting.

**Why use ColumnTransformer?**  
To apply different transformations to different feature groups.

**Why use a Pipeline?**  
To combine preprocessing and modeling into a reproducible workflow.

**Why fit the preprocessor only on training data?**  
To prevent information from the test set influencing preprocessing.

**Should `duration` be included in pre-contact predictions?**  
No, because its value is not yet available.

## 15. Key Takeaway

Preprocessing converts our numerical and categorical inputs into a model-ready representation.

For our Bank Marketing project, we will use `ColumnTransformer` to scale numerical features and one-hot encode categorical features, fit transformations on training data only, and preserve the same transformations for test data and later model evaluation.
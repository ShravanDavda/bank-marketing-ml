# Feature Engineering — Bank Marketing ML

## 1. What Is Feature Engineering?

Feature engineering is the process of creating, transforming, or deriving input features from existing data to represent useful information for a machine learning model.

A dataset may contain useful information that is not explicitly represented in the most convenient form. Feature engineering can make that information easier for a model to use.

For example, our dataset contains `pdays`, which represents the number of days since a previous campaign contact. A value of `-1` indicates that the customer was not previously contacted.

We can create a new binary feature called `previously_contacted` to represent this information explicitly.

**Our objective:** Create only features that have a clear meaning and a justified role in the Bank Marketing project.

---

## 2. Why Do We Need Feature Engineering?

Feature engineering can help us:

- Represent useful information more clearly.
- Capture meaningful characteristics of customers.
- Make relationships easier for a model to learn.
- Incorporate domain knowledge into the dataset.
- Prepare a suitable set of inputs for model development.

However, creating more features does not automatically improve model performance. Every new feature should have a clear justification and should eventually be evaluated.

---

## 3. Feature Engineering vs Data Preprocessing

These two stages are related but serve different purposes.

| Feature engineering | Data preprocessing |
|---|---|
| Creates or transforms useful input features | Makes data suitable for a particular algorithm |
| Can incorporate domain knowledge | Handles operations such as encoding and scaling |
| Example: create `previously_contacted` from `pdays` | Example: one-hot encode `job` |
| Example: derive a meaningful indicator from existing columns | Example: standardize numerical features |

In our project, we will handle categorical encoding and numerical scaling in Lesson 7 — Data Preprocessing.

We should not duplicate those operations in this lesson.

---

## 4. Our EDA Findings and Feature Decisions

During EDA, we observed:

- The target is imbalanced, with approximately 88.3% `no` and 11.7% `yes`.
- `balance`, `duration`, and `campaign` have visibly right-skewed distributions.
- Subscription rates vary across several categorical groups.
- `pdays` has a special value of `-1` representing no previous contact.

These findings help us identify questions worth investigating. They do not automatically justify creating or transforming every feature.

Our initial decisions are:

| Feature or idea | Decision | Reason |
|---|---|---|
| `previously_contacted` | Create | Explicitly represents previous-contact status |
| Transform `balance` | Defer | Skewness alone does not justify a transformation |
| Transform `campaign` | Defer | Evaluate whether a derived representation is useful |
| Encode categorical features | Lesson 7 | Preprocessing operation |
| Scale numerical features | Lesson 7 | Preprocessing operation |
| Use `duration` | Review prediction timing | May not be available before a customer is contacted |

---

## 5. Creating a Binary Feature From `pdays`

### 5.1 Understand the original feature

The `pdays` feature represents the number of days since the customer was previously contacted in an earlier campaign.

In this dataset:

- `pdays = -1`: the customer was not previously contacted.
- `pdays != -1`: a previous contact was recorded.

The original feature contains both the time-related information and a special indicator for customers without a previous contact.

### 5.2 Derive `previously_contacted`

We create a new feature:

```python
df["previously_contacted"] = (
    df["pdays"] != -1
).astype(int)
```

The expression `df["pdays"] != -1` produces Boolean values:

- `True` when `pdays` is not `-1`.
- `False` when `pdays` equals `-1`.

The `.astype(int)` conversion changes:

- `True` to `1`
- `False` to `0`

The resulting feature has a simple interpretation:

| `pdays` condition | `previously_contacted` |
|---|---:|
| `pdays = -1` | 0 |
| `pdays != -1` | 1 |

This is a derived binary feature.

### 5.3 Why might it be useful?

The new feature explicitly represents whether previous contact occurred.

Some models may use this indicator differently from the original numerical value of `pdays`. However, the feature's usefulness must be assessed through model evaluation rather than assumed.

### 5.4 Should we remove `pdays`?

Not at this stage.

We will initially retain both:

- `pdays`
- `previously_contacted`

They contain related information, but they represent it differently.

Later, model comparison can help us investigate whether both are useful. We should avoid making an unsupported feature-selection decision now.

---

## 6. Verifying the New Feature

After creating a feature, we should verify that it was created correctly.

```python
df[["pdays", "previously_contacted"]].head(10)
```

Inspect the original values alongside the derived indicator.

Next, inspect the counts:

```python
df["previously_contacted"].value_counts()
```

We can also verify the permitted values:

```python
assert df["previously_contacted"].isin([0, 1]).all()
```

And verify the relationship with the original feature:

```python
assert (
    df["previously_contacted"]
    == (df["pdays"] != -1).astype(int)
).all()
```

If these checks pass, the feature matches the intended definition.

Creating a new feature should not remove existing columns or change the target variable.

---

## 7. Feature Availability and Data Leakage

Feature engineering must consider when a prediction will be made.

A feature may be present in the historical dataset but unavailable at the intended prediction time.

For example, `duration` represents the duration of the last contact in seconds.

If our goal is to predict whether a customer will subscribe **before contacting them**, the duration of that contact is not yet known. Using it would make the model unsuitable for that prediction scenario.

If the goal is to predict an outcome after the call, duration may be available, but the model would serve a different purpose.

Therefore, before finalizing our input features, we must define the intended prediction point and exclude features that would not legitimately be available then.

This is a feature-validity and modeling decision, not simply a numerical transformation.

---

## 8. Why We Are Not Creating Many More Features Yet

Our EDA showed skewness in `balance`, `duration`, and `campaign`.

However, skewness does not automatically mean that we should apply logarithmic transformations, bin numerical variables, or remove extreme values.

Such changes should be justified by a clear objective and evaluated appropriately.

For this project, we will keep feature engineering focused. We can consider additional transformations later if the analysis or model evaluation provides a good reason.

Similarly, we will not automatically create features from every pair of columns or perform complex feature-selection procedures without evidence that they are necessary.

---

## 9. Common Mistakes

1. **Creating features without a clear purpose.**  
   Every derived feature should represent useful information.

2. **Assuming more features always improve performance.**  
   Additional features may provide little value or introduce unnecessary complexity.

3. **Confusing feature engineering with preprocessing.**  
   Creating an indicator is feature engineering; encoding categories and scaling are preprocessing operations.

4. **Removing the original feature immediately.**  
   Retain `pdays` while we evaluate the new indicator.

5. **Treating every unusual value as invalid.**  
   The value `pdays = -1` has a specific meaning.

6. **Transforming every skewed feature automatically.**  
   Skewness is a reason to investigate, not an automatic instruction to transform.

7. **Ignoring prediction-time availability.**  
   Features must be available when the model is intended to make its prediction.

8. **Claiming improvement without evaluation.**  
   A feature is not proven useful simply because it has been created.

---

## 10. Connection to Our Notebook

This study material corresponds to:

`notebooks/bank_marketing_ml.ipynb`

Section:

`7. Feature Engineering`

Planned subsections:

- 7.1 Previous Contact Indicator
- 7.2 Feature Verification
- 7.3 Feature Engineering Decisions

The notebook contains the implementation, actual outputs, and concise interpretations. This document contains the fuller explanations for learning and revision.

---

## 11. Quick Revision

**Q1. What is feature engineering?**  
Creating or transforming input features to represent useful information for a model.

**Q2. What is the difference between feature engineering and preprocessing?**  
Feature engineering creates or transforms features; preprocessing prepares data for an algorithm through operations such as encoding and scaling.

**Q3. What does `previously_contacted = 0` mean?**  
The customer was not previously contacted according to the dataset's `pdays` convention.

**Q4. Why do we retain `pdays`?**  
We have not yet established that removing it would improve the project.

**Q5. Does adding a feature guarantee better performance?**  
No. Its usefulness must be evaluated.

**Q6. Why must we review `duration`?**  
It may not be available when predicting subscription before contacting the customer.

**Q7. What is the central rule of feature engineering?**  
Create meaningful features, verify them, and evaluate their usefulness rather than adding complexity without evidence.

---

## 12. Key Takeaway

Feature engineering should be deliberate and evidence-based.

For our Bank Marketing project, we create `previously_contacted` from `pdays` because it expresses previous-contact status clearly. We preserve the original feature and defer other transformations until there is a justified reason to introduce them.
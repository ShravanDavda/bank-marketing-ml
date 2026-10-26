# Data Cleaning — Bank Marketing ML

## 1. Purpose of Data Cleaning

Data cleaning is the process of identifying and addressing data-quality problems before using a dataset for analysis or machine learning.

Common data-quality issues include:

- Missing values
- Duplicate records
- Invalid or unexpected values
- Incorrect data types
- Inconsistent categorical labels
- Placeholder values that require special interpretation

The goal is not to change every unusual value. The goal is to make the dataset reliable while preserving valid information.

**Our project objective:** Inspect the Bank Marketing dataset and determine which cleaning operations, if any, are justified before exploratory data analysis and model development.

---

## 2. Our Dataset Before Cleaning

We use `bank-full.csv` as our primary dataset.

| Property | Initial finding |
|---|---|
| Rows | 45,211 |
| Columns | 17 |
| Input features | 16 |
| Target | `y` |
| Missing values detected | 0 |
| Completely duplicated rows | 0 |

These findings come from the initial dataset inspection. They do not guarantee that every value is valid or consistent.

For example, a dataset can have zero null values but still contain literal `"unknown"` categories, invalid values, or inconsistent labels.

---

## 3. Data Cleaning Workflow

We follow an evidence-based process:

```text
Original Dataset
       ↓
Inspect Data Quality
       ↓
Identify Potential Issues
       ↓
Understand Their Meaning
       ↓
Choose Appropriate Action
       ↓
Apply Only Justified Changes
       ↓
Verify the Result
```

For every potential issue, ask:

1. What did we observe?
2. Why might it be a problem?
3. Is the value actually invalid?
4. What action is justified?
5. How will we verify the result?

This prevents unnecessary transformations and accidental loss of useful information.

---

## 4. Missing Values

### 4.1 What are missing values?

A missing value indicates that a value is absent or unavailable.

Examples include:

- A missing customer age
- An unavailable account balance
- An empty value in a categorical column

Missing values are commonly represented as `NaN` or `None` in Pandas.

### 4.2 Detect missing values

```python
df.isnull().sum()
```

This returns the number of missing values in each column.

To calculate the total:

```python
df.isnull().sum().sum()
```

The first `sum()` calculates missing values per column. The second adds those counts together.

### 4.3 How do we handle missing values?

The appropriate method depends on the problem.

| Method | When it may be appropriate |
|---|---|
| Mean imputation | Filling missing numerical values when the mean is suitable |
| Median imputation | Filling missing numerical values when the median is more appropriate |
| Mode imputation | Filling missing categorical values with a frequent category |
| Explicit missing category | Preserving missingness as a distinct category when justified |
| Row removal | Removing records only when the loss of information is acceptable |

These are possible techniques, not mandatory operations.

**Our dataset:** The initial null check found zero missing values. Therefore, we do not need null-value imputation based on that check.

### Important distinction

A literal string such as `"unknown"` is not necessarily recognized by Pandas as a missing value.

Therefore:

```python
df.isnull().sum()
```

does not tell us whether literal `"unknown"` categories exist.

We inspect categorical values separately.

---

## 5. Duplicate Records

### 5.1 What are duplicate records?

Duplicate records are rows that repeat the same values across all columns.

They may arise from repeated data collection, file merging, or data-entry processes.

### 5.2 Detect duplicate rows

```python
df.duplicated().sum()
```

This counts duplicate rows after their first occurrence.

To inspect them:

```python
df[df.duplicated()]
```

To remove duplicates when removal is justified:

```python
df = df.drop_duplicates()
```

### 5.3 Should we always remove duplicates?

No.

Two records with identical values are not automatically proof that the same real-world customer was recorded twice. The appropriate decision depends on what one row represents and how the dataset was collected.

**Our dataset:** The initial check found zero completely duplicated rows. Therefore, duplicate removal is not currently required based on that check.

---

## 6. Data Types and Numerical Ranges

### 6.1 Why inspect data types?

Data types influence how Python processes values and how preprocessing should be designed.

Inspect them using:

```python
df.dtypes
```

Or:

```python
df.info()
```

Our dataset contains numerical features and categorical features.

A column that contains numbers is not automatically a continuous measurement. For example, `day` represents a day of the month, while `campaign` represents a count.

### 6.2 Inspect numerical ranges

```python
df[numerical_features].describe().T
```

This displays descriptive statistics for numerical features.

Look for:

- Unexpected negative values
- Implausible minimum or maximum values
- Unusual ranges
- Values that need domain-specific interpretation

A value that looks unusual is a reason to investigate, not automatic proof of an error.

### 6.3 Project-specific checks

```python
print("Age below 18:", (df["age"] < 18).sum())
print("Negative balance:", (df["balance"] < 0).sum())
print("Negative duration:", (df["duration"] < 0).sum())
print("Negative campaign:", (df["campaign"] < 0).sum())
print("pdays equal to -1:", (df["pdays"] == -1).sum())
```

Interpret these checks carefully:

- A negative account balance can be valid.
- A negative contact duration would require investigation.
- A negative campaign count would require investigation.
- The value `-1` in `pdays` has a special meaning in this dataset.

We should not delete records solely because a numerical value is negative.

---

## 7. Understanding the Special Value `pdays = -1`

The `pdays` feature represents the number of days since a customer was previously contacted in an earlier campaign.

In this dataset, `-1` represents customers who were not previously contacted.

Therefore, `-1` is a meaningful placeholder rather than automatically an invalid number.

We can inspect its frequency:

```python
df["pdays"].value_counts().sort_index()
```

We can also count these records:

```python
(df["pdays"] == -1).sum()
```

### What should we do?

For now, preserve this value.

During exploratory data analysis and feature engineering, we can evaluate whether it would be useful to create an additional feature indicating whether a previous contact occurred.

We should make that decision based on the project requirements and analysis, not automatically transform the column.

---

## 8. Categorical Consistency

### 8.1 What are categorical inconsistencies?

Categorical inconsistencies occur when the same logical category is represented in different ways.

For example:

```text
admin
Admin
admin
```

These may represent the same category but appear as different values to a computer.

Whitespace can create a similar issue:

```text
married
married 
```

### 8.2 Inspect category frequencies

```python
for column in categorical_features + [target]:
    print(f"\n--- {column} ---")
    print(df[column].value_counts(dropna=False))
```

This displays the frequency of each value, including null values.

Inspect the output for:

- Unexpected category labels
- Inconsistent capitalization
- Extra whitespace
- Literal `"unknown"` values
- Categories that require domain interpretation

### 8.3 Check surrounding whitespace

```python
for column in categorical_features + [target]:
    values = df[column].dropna().astype(str)

    inconsistent_whitespace = values.ne(
        values.str.strip()
    ).sum()

    print(
        f"{column}: "
        f"{inconsistent_whitespace} values with surrounding whitespace"
    )
```

This counts non-null values with leading or trailing whitespace.

If an inconsistency is found, we should determine whether normalization is appropriate before changing the data.

### 8.4 Normalize values only when justified

For example, if inspection confirms that surrounding whitespace is accidental, we could normalize the affected columns:

```python
df[categorical_features + [target]] = (
    df[categorical_features + [target]]
    .apply(lambda column: column.str.strip())
)
```

This code removes surrounding whitespace from string values in the selected columns.

**Do not run this transformation automatically.** First inspect the data. In our project, we should only apply it if it addresses a confirmed issue.

Similarly, changing all categories to lowercase may be unnecessary if their existing labels are already consistent.

---

## 9. Understanding `"unknown"` Values

A categorical value such as `"unknown"` may indicate that information was unavailable or not recorded.

It is different from a null value.

For example:

```text
education = unknown
```

may mean the customer's education category is not known.

Possible approaches include:

- Keep `"unknown"` as a distinct category.
- Investigate whether the category represents missing information.
- Consider alternative handling if the project evidence justifies it.

We should not automatically delete these records or replace `"unknown"` with the most frequent category.

Keeping the category initially preserves information until we understand its meaning.

---

## 10. Outliers and Invalid Values

### 10.1 What is an outlier?

An outlier is an observation that differs substantially from the rest of the data.

For example, an unusually large account balance may be an outlier.

However, an outlier is not necessarily an error.

### 10.2 How should we handle outliers?

Possible actions include:

- Preserve valid extreme values.
- Correct confirmed data-entry errors when the correct value is known.
- Remove invalid records when there is a justified reason.
- Consider transformations or robust modeling techniques if extreme values affect the analysis.

Do not automatically remove every value outside a chosen statistical threshold.

### 10.3 What belongs in EDA?

Data cleaning focuses on identifying and correcting genuine data-quality problems.

EDA will help us study distributions, unusual observations, and relationships between variables in greater detail.

We should not make an irreversible cleaning decision based only on a summary statistic.

---

## 11. Data Leakage Awareness

Data leakage occurs when information unavailable at prediction time influences model training or evaluation inappropriately.

For example, if a model uses information that directly reveals the outcome it is supposed to predict, its measured performance may be misleading.

For this project, the `duration` feature deserves special attention. It represents the duration of the last contact in seconds. Depending on when a prediction is intended to be made, that information may only become available after the call has ended.

If the intended use is to prioritize customers before contacting them, `duration` would not be available at prediction time and should not be used as an input for that scenario.

We must define the intended prediction point before deciding whether to retain this feature.

This is a modeling and data-validity decision, not simply a missing-value cleaning operation.

---

## 12. Cleaning Decisions for Our Project

Based on the initial inspection, our current position is:

| Potential issue | Current decision |
|---|---|
| Null values | No imputation currently required |
| Exact duplicate rows | No removal currently required |
| `pdays = -1` | Preserve its documented meaning |
| Negative account balances | Do not remove automatically |
| Literal `"unknown"` categories | Inspect and interpret |
| Categorical inconsistencies | Audit before normalizing |
| Extreme numerical values | Investigate before treating |
| `duration` availability | Review against the intended prediction time |

Some of these are confirmed initial findings, while others are checks we still need to perform.

We must not claim that the dataset is completely clean until the relevant checks have been run and reviewed.

---

## 13. Verification After Cleaning

After applying any justified transformation, repeat the relevant checks.

For example:

```python
df.isnull().sum()
```

```python
df.duplicated().sum()
```

```python
df.dtypes
```

Also compare the dataset shape before and after any operation that removes rows:

```python
df.shape
```

Verification helps us confirm that a transformation achieved its intended purpose without introducing new problems.

For more complex cleaning, it is useful to retain an unchanged copy of the original data:

```python
df = pd.read_csv("../bank-full.csv", sep=";")
df_clean = df.copy()
```

Then apply justified transformations to `df_clean` while preserving `df` for comparison.

Do not create multiple copies unnecessarily for every simple inspection; use a separate cleaned DataFrame when it improves clarity and reproducibility.

---

## 14. Common Mistakes

1. **Assuming zero nulls means perfect data.**  
   Other data-quality issues may still exist.

2. **Treating `"unknown"` as a null automatically.**  
   It is a literal category unless explicitly represented otherwise.

3. **Deleting every negative value.**  
   Negative balances can be legitimate, and `pdays = -1` has a documented meaning.

4. **Removing every outlier.**  
   Extreme values can be valid observations.

5. **Removing duplicates without understanding the records.**  
   Exact duplicate rows do not always prove duplicate real-world entities.

6. **Normalizing every categorical column automatically.**  
   Transformations should solve an identified problem.

7. **Changing the dataset without checking the result.**  
   Always verify important transformations.

8. **Ignoring prediction-time information.**  
   A feature can be present in the dataset but unavailable at the time the model is intended to make its prediction.

9. **Claiming checks passed without running them.**  
   Record actual results, not assumed results.

10. **Mixing data cleaning with model preprocessing.**  
    Data cleaning addresses data-quality problems; encoding and scaling are generally handled in the preprocessing pipeline later.

---

## 15. Connection to Our Notebook

This document corresponds to:

```text
5. Data Cleaning
   5.1 Data Quality Audit
   5.2 Categorical Value Inspection
   5.3 Numerical Range Inspection
   5.4 Text Consistency Checks
   5.5 Cleaning Decisions
   5.6 Verification
```

The notebook should contain the relevant checks, actual outputs, and concise interpretations. This MD contains the fuller explanations for learning and revision.

---

## 16. Quick Revision

**Q1. What is data cleaning?**  
Identifying and addressing data-quality problems before analysis and modeling.

**Q2. Why can a dataset have zero missing values but still need investigation?**  
It may contain invalid values, literal placeholders, inconsistent labels, or other problems.

**Q3. How do we check duplicate rows?**  
`df.duplicated().sum()`

**Q4. Should every negative numerical value be deleted?**  
No. Its meaning depends on the feature and domain.

**Q5. What does `pdays = -1` mean in our dataset?**  
The customer was not previously contacted.

**Q6. Is `"unknown"` the same as a null value?**  
Not necessarily. It may be a literal categorical value.

**Q7. Should we always remove outliers?**  
No. We should determine whether they are errors or valid observations.

**Q8. What is data leakage?**  
Using information that would not legitimately be available for the intended prediction or evaluation.

**Q9. Why is `duration` important to investigate?**  
Its availability depends on when the prediction is intended to occur.

**Q10. What is the central rule of data cleaning?**  
Inspect first, justify the change, apply it, and verify the result.

---

## 17. Key Takeaway

Data cleaning is not about making every column look perfect. It is about preserving valid information and correcting genuine data-quality problems.

For the Bank Marketing project, we will base every cleaning operation on evidence from the actual dataset and the intended use of the model.
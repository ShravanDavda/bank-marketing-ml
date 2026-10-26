# Dataset Understanding — Bank Marketing ML

## 1. Purpose

Dataset understanding is the process of inspecting a dataset before performing data cleaning, exploratory data analysis (EDA), feature engineering, or model training.

It helps us understand:

- The size and structure of the dataset.
- What each column represents.
- The types of variables available.
- Whether missing values or duplicate records exist.
- The distribution of the target variable.
- Any unusual values that require domain knowledge.

**In our project:** We need to understand the Bank Marketing dataset before using it to predict whether a customer will subscribe to a term deposit.

---

## 2. Our Dataset

We have two dataset files:

| File | Rows | Columns |
|---|---:|---:|
| `bank.csv` | 4,521 | 17 |
| `bank-full.csv` | 45,211 | 17 |

Both datasets contain the same 17-column schema.

We will use `bank-full.csv` as the primary dataset because it contains more records. We will keep `bank.csv` available but will not combine the two datasets for model training.

### Dataset structure

- **Rows:** Individual records in the dataset.
- **Columns:** Variables describing each record.
- **Features (`X`):** Input variables used for prediction.
- **Target (`y`):** The variable we want to predict.

Our primary dataset contains 45,211 rows and 17 columns: 16 input features and one target variable.

---

## 3. Loading the Dataset

We use Pandas to load the CSV file into a DataFrame.

```python
import pandas as pd

df = pd.read_csv("../bank-full.csv", sep=";")
```

### What is Pandas?

Pandas is a Python library for working with structured data. Its DataFrame object represents data in a table-like format.

### Why use `sep=";"`?

The Bank Marketing CSV file uses semicolons (`;`) to separate its fields. Specifying the separator ensures that Pandas reads the columns correctly.

If we omit the separator when the file is semicolon-delimited, the data may be parsed incorrectly.

### What is a DataFrame?

A DataFrame is a two-dimensional table containing rows and columns. We use `df` as the variable name to store our dataset.

---

## 4. Basic Dataset Inspection

### 4.1 Dataset shape

```python
df.shape
```

The result is a tuple:

```text
(45211, 17)
```

The first number represents rows; the second represents columns.

**Why it matters:** It gives us an immediate overview of the dataset's size.

### 4.2 First five records

```python
df.head()
```

This displays the first five rows.

We can display more rows with:

```python
df.head(10)
```

**Why it matters:** It allows us to check the column names, example values, and whether the data appears to have loaded correctly.

### 4.3 Last five records

```python
df.tail()
```

This displays the final five rows. It can help us inspect the end of the dataset.

### 4.4 Column names

```python
df.columns.tolist()
```

This returns the column names as a Python list.

The 17 columns are:

```text
age, job, marital, education, default, balance,
housing, loan, contact, day, month, duration,
campaign, pdays, previous, poutcome, y
```

---

## 5. Understanding Data Types

Different types of data require different preprocessing methods.

### 5.1 Numerical features

The numerical features in our dataset are:

- `age` — customer age
- `balance` — account balance
- `day` — day of the month of the last contact
- `duration` — duration of the last contact in seconds
- `campaign` — number of contacts during the current campaign
- `pdays` — days since a previous campaign contact, with a special value of `-1`
- `previous` — number of contacts before the current campaign

These seven features are represented numerically in the dataset.

### 5.2 Categorical features

The categorical features are:

- `job` — job category
- `marital` — marital status
- `education` — education level
- `default` — credit in default status
- `housing` — housing loan status
- `loan` — personal loan status
- `contact` — communication method
- `month` — month of the last contact
- `poutcome` — outcome of the previous campaign

These nine features represent categories or labels rather than quantities on a meaningful numerical scale.

For example, `month` may be stored as text, and a month number should not automatically be treated as a continuous measurement.

### 5.3 Target variable

The target column is:

```text
y
```

Its two possible values are:

- `yes` — the customer subscribed.
- `no` — the customer did not subscribe.

The target is categorical, so this is a binary classification task.

### 5.4 Inspecting data types

```python
df.dtypes
```

This displays the data type of each column.

We can also inspect the full DataFrame information:

```python
df.info()
```

`info()` provides:

- Number of rows.
- Column names.
- Non-null counts.
- Data types.
- Approximate memory usage.

In Pandas, categorical columns may initially appear as `object`. We will handle their encoding during preprocessing rather than changing their types unnecessarily at this stage.

---

## 6. Checking Missing Values

Missing values are values that are absent or recorded as null.

They can cause problems for some preprocessing steps and machine learning algorithms, so we should inspect them before training.

### 6.1 Missing values per column

```python
df.isnull().sum()
```

This calculates the number of missing values in each column.

### 6.2 Total missing values

```python
df.isnull().sum().sum()
```

The first `.sum()` counts missing values within each column. The second `.sum()` adds those column totals together.

For our `bank-full.csv` dataset, the total is **0**.

Therefore, no missing-value imputation is required for the null values detected by this check.

**Important:** Null checks do not automatically identify every unusual or placeholder value. Those require separate inspection and domain knowledge.

---

## 7. Checking Duplicate Records

Duplicate records are rows that repeat the same values across all columns.

We can check for them using:

```python
df.duplicated().sum()
```

This counts duplicate rows after their first occurrence.

For our `bank-full.csv` dataset, the result is **0**.

Therefore, we do not need to remove any completely duplicated rows based on this check.

**Important:** Two customers having identical feature values does not necessarily prove that they are duplicate records. Duplicate detection should be interpreted in the context of the data.

---

## 8. Understanding the Target Distribution

Before training a classifier, we should inspect how frequently each target class occurs.

### 8.1 Count each class

```python
df["y"].value_counts()
```

For our primary dataset:

| Target | Count |
|---|---:|
| `no` | 39,922 |
| `yes` | 5,289 |

### 8.2 Calculate class percentages

```python
df["y"].value_counts(normalize=True) * 100
```

The approximate distribution is:

- `no`: 88.3%
- `yes`: 11.7%

### 8.3 What is class imbalance?

Class imbalance occurs when the target classes have substantially different frequencies.

Our dataset is imbalanced because non-subscriptions are much more common than subscriptions.

This matters because a model can achieve apparently high accuracy while performing poorly at identifying customers who actually subscribe.

For example, a model that always predicts `no` would achieve approximately 88.3% accuracy on this dataset, yet it would identify none of the actual `yes` cases.

Therefore, later we will evaluate models using accuracy alongside precision, recall, F1-score, and other suitable metrics.

We do not need to apply an imbalance-handling technique automatically. We should first evaluate the models and determine whether any additional technique is justified.

---

## 9. Descriptive Statistics

We use `describe()` to summarize numerical variables.

```python
df.describe()
```

By default, this summarizes numerical columns using statistics such as:

| Statistic | Meaning |
|---|---|
| `count` | Number of non-null observations |
| `mean` | Arithmetic average |
| `std` | Standard deviation, a measure of spread |
| `min` | Minimum observed value |
| `25%` | First quartile |
| `50%` | Median |
| `75%` | Third quartile |
| `max` | Maximum observed value |

For example, these statistics can help us inspect the range and distribution of `age`, `balance`, and `duration`.

We can also summarize categorical columns:

```python
df.describe(include="object")
```

This typically reports the number of non-null observations, number of unique values, most frequent value, and its frequency for object-type columns.

**Important:** Summary statistics are an initial inspection tool. Detailed analysis of distributions, outliers, and relationships belongs in EDA.

---

## 10. Checking Unique Values

We can count the number of distinct values in each column:

```python
df.nunique()
```

Examples of expected cardinalities include:

- `marital` — 3 categories
- `education` — 4 categories
- `default` — 2 categories
- `housing` — 2 categories
- `loan` — 2 categories
- `month` — 12 categories
- `poutcome` — 4 categories
- `y` — 2 classes

Here, **cardinality** means the number of distinct values in a feature.

This information helps us understand the categorical variables and plan preprocessing. It does not, by itself, determine which encoding method is best.

---

## 11. Important Dataset-Specific Observation: `pdays`

The `pdays` feature represents the number of days since the customer was previously contacted in an earlier campaign.

A value of `-1` is used to indicate that the customer was not previously contacted.

Therefore, we should not automatically treat `-1` as a data error.

This is an example of why data understanding must include the meaning of the variables, not just their numerical ranges.

During later analysis, we can examine how this special value is distributed and decide whether any feature transformation is justified.

---

## 12. Dataset Understanding vs EDA

These are related but different stages.

### Dataset Understanding asks:

**What is in the dataset?**

Examples:

- How many rows and columns exist?
- What does each feature represent?
- Which variables are numerical or categorical?
- Are missing values or duplicate rows present?
- How frequently does each target class occur?

### Exploratory Data Analysis asks:

**What patterns and relationships exist in the data?**

Examples:

- How is customer age distributed?
- How does subscription vary across job categories?
- How do numerical features differ between subscribers and non-subscribers?
- Which patterns deserve further investigation?

The workflow is:

```text
Dataset Understanding
        ↓
Data Quality Checks
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Development
```

We may identify preliminary issues during dataset understanding, but we should reserve detailed visual and relationship analysis for EDA.

---

## 13. Common Mistakes to Avoid

1. **Loading the CSV with the wrong separator.**  
   Our dataset uses semicolons, so specify `sep=";"`.

2. **Assuming all numbers are continuous measurements.**  
   A numerical representation does not automatically make a feature continuous or meaningful as a measurement.

3. **Treating every unusual value as an error.**  
   The value `-1` in `pdays` has a specific meaning.

4. **Assuming no missing values means the data is perfect.**  
   Placeholder values, inconsistent labels, invalid values, and potential leakage require separate checks.

5. **Deleting duplicates without understanding the records.**  
   Duplicate checks must be interpreted in context.

6. **Using accuracy alone for an imbalanced target.**  
   A model can perform poorly on the minority class despite high overall accuracy.

7. **Mixing the two dataset files.**  
   Use `bank-full.csv` consistently as the primary dataset unless a documented reason requires otherwise.

8. **Performing every cleaning operation automatically.**  
   Inspect first, then apply only justified transformations.

9. **Putting too much theory into the notebook.**  
   Keep detailed explanations in the study material and concise explanations in the notebook.

---

## 14. Our Project Findings

The initial inspection of `bank-full.csv` established the following:

| Check | Finding |
|---|---|
| Rows | 45,211 |
| Columns | 17 |
| Input features | 16 |
| Target | `y` |
| Target classes | `yes`, `no` |
| Numerical features | 7 |
| Categorical features | 9 |
| Missing values detected by null checks | 0 |
| Completely duplicated rows | 0 |
| Target distribution | Approximately 88.3% `no`, 11.7% `yes` |
| Special value to investigate | `pdays = -1` |

These findings describe the initial dataset. They do not yet establish whether every feature is useful, whether outliers require treatment, or which model will perform best.

---

## 15. Connection to Our Notebook

This study material corresponds to these notebook sections:

```text
2. Import Libraries
3. Load Dataset
4. Data Understanding
    4.1 Dataset Shape
    4.2 Dataset Information
    4.3 Column Names
    4.4 Feature Types
    4.5 Missing Values
    4.6 Duplicate Records
    4.7 Target Distribution
    4.8 Descriptive Statistics
    4.9 Unique Values
    4.10 Important Observations
```

The notebook should contain the code, relevant outputs, and short interpretations. This document contains the fuller explanations for learning and revision.

---

## 16. Quick Revision

**Q1. What is a DataFrame?**  
A two-dimensional, table-like data structure provided by Pandas.

**Q2. What does `df.shape` return?**  
A tuple containing the number of rows and columns.

**Q3. What does `df.info()` show?**  
Column names, non-null counts, data types, and memory information.

**Q4. How do we check missing values?**  
Use `df.isnull().sum()`.

**Q5. How do we check duplicate rows?**  
Use `df.duplicated().sum()`.

**Q6. What is class imbalance?**  
A substantial difference in the frequencies of target classes.

**Q7. Why is our target imbalanced?**  
Approximately 88.3% of records have `y = no`, compared with 11.7% having `y = yes`.

**Q8. Why is `pdays = -1` important?**  
It represents customers who were not previously contacted, rather than automatically indicating an error.

**Q9. What is the difference between dataset understanding and EDA?**  
Dataset understanding inspects the structure and basic quality; EDA investigates distributions and relationships.

**Q10. What comes next?**  
Data Cleaning: determining whether any transformations or corrections are actually necessary.

---

## 17. Key Takeaway

Dataset understanding is not about changing the data immediately. It is about establishing what the data contains, checking its basic quality, and identifying issues that deserve further investigation.

For our project, this gives us a reliable foundation for cleaning, EDA, preprocessing, and model development.
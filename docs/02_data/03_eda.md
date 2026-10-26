# Exploratory Data Analysis (EDA) — Bank Marketing ML

## 1. What Is EDA?

Exploratory Data Analysis (EDA) is the process of examining a dataset using descriptive statistics and visualizations to understand its distributions, patterns, relationships, and potential issues.

EDA helps us understand the data before deciding how to engineer features or build machine learning models.

In our Bank Marketing project, EDA helps us investigate customer characteristics and how subscription outcomes vary across customer and campaign attributes.

## 2. Why Do We Need EDA?

EDA helps us:

- Understand the distribution of the target variable.
- Examine numerical feature distributions.
- Understand categorical feature frequencies.
- Compare subscription rates across customer groups.
- Identify unusual patterns that deserve investigation.
- Form evidence-based hypotheses for later modeling.

EDA does not prove causation and does not automatically identify the best predictive model.

## 3. EDA vs Dataset Understanding

**Dataset Understanding** asks:

> What is contained in the dataset?

Examples:
- Number of rows and columns
- Feature names and data types
- Missing values and duplicate rows
- Target classes

**EDA** asks:

> What patterns and relationships can we observe?

Examples:
- How is age distributed?
- How does subscription vary by job?
- Are numerical features skewed?
- How does the subscription rate differ across contact methods?

## 4. Visualization Libraries

### Matplotlib

Matplotlib is a Python plotting library used to create and customize charts.

```python
import matplotlib.pyplot as plt
```

### Seaborn

Seaborn provides statistical visualization functions built on Matplotlib.

```python
import seaborn as sns

sns.set_theme(style="whitegrid")
```

We use these libraries to create readable charts with titles, labels, and appropriate figure sizes.

## 5. Target Distribution

Our target variable is `y`, with two classes:

- `yes`: the customer subscribed.
- `no`: the customer did not subscribe.

We inspect class counts using:

```python
df["y"].value_counts()
```

And percentages using:

```python
df["y"].value_counts(normalize=True) * 100
```

A count plot can display the frequency of each class.

### Why it matters

The target is imbalanced: approximately 88.3% of records are `no` and 11.7% are `yes` in our primary dataset.

This matters during model evaluation because a classifier can achieve high accuracy while failing to identify subscribers.

The distribution alone does not tell us which model will perform best.

## 6. Numerical Feature Distributions

Numerical features include:

- `age`
- `balance`
- `day`
- `duration`
- `campaign`
- `pdays`
- `previous`

Histograms help us examine how numerical values are distributed.

Example:

```python
sns.histplot(data=df, x="age", bins=30)
plt.title("Distribution of Customer Age")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()
```

### What to observe

- Where values are concentrated
- Spread and range
- Skewness
- Potentially unusual values
- Whether transformations or further investigation may be useful

An unusual value is not automatically an error. For example, negative account balances may be valid.

We will focus on a small set of meaningful numerical features rather than plotting every feature without purpose.

## 7. Categorical Feature Distributions

Categorical features include:

- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`
- `contact`
- `month`
- `poutcome`

A count plot displays the number of observations in each category.

Example:

```python
sns.countplot(data=df, x="job")
plt.title("Customer Distribution by Job")
plt.xlabel("Job")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

### What to observe

- Frequent and rare categories
- Differences in category sizes
- Categories that may require careful interpretation

Category frequency is not the same as subscription rate. A common category may have more subscribers simply because it contains more customers.

## 8. Comparing Features With the Target

We want to understand how the subscription outcome varies across groups.

For a categorical feature, calculate the percentage of customers with `y = yes` within each category.

```python
subscription_rate = (
    df.assign(subscribed=(df["y"] == "yes").astype(int))
      .groupby("job")["subscribed"]
      .mean()
      .mul(100)
      .sort_values(ascending=False)
)
```

This produces the observed subscription rate for each job category.

We can visualize it:

```python
subscription_rate.plot(kind="bar", figsize=(10, 5))

plt.title("Subscription Rate by Job")
plt.xlabel("Job")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

### Why use rates?

Categories may have very different numbers of customers. Rates allow us to compare the proportion of subscribers rather than raw counts alone.

### Important limitation

These are observed associations, not proof that a job category causes subscription. Other features may influence the observed differences.

## 9. Numerical Features and Subscription

We can compare numerical distributions across the target classes.

Example:

```python
sns.boxplot(data=df, x="y", y="age")

plt.title("Age by Subscription Outcome")
plt.xlabel("Subscription")
plt.ylabel("Age")
plt.show()
```

A box plot summarizes the median, quartiles, and potential extreme observations.

It helps us compare the distribution of age between subscribers and non-subscribers.

We can use similar plots for selected features such as `balance` and `duration`, choosing suitable scales when distributions are highly skewed.

## 10. Important Project-Specific Consideration: Duration

The `duration` feature records the duration of the last contact in seconds.

It may show a strong relationship with the subscription outcome, but we must consider when a prediction is intended to occur.

If the model must prioritize customers before contacting them, call duration is not yet available and should not be used as an input for that prediction scenario.

EDA can reveal a relationship; it cannot determine by itself whether a feature is legitimately available at prediction time.

## 11. Correlation and Categorical Variables

Correlation can help summarize relationships between numerical variables.

However, ordinary numerical correlation is not automatically meaningful for categorical labels, and correlation does not establish causation.

We will only add correlation analysis when it answers a useful project question. We do not need a large correlation section merely to make the notebook look advanced.

## 12. How to Interpret EDA Results

For every visualization, explain:

1. What was plotted?
2. What pattern is actually visible?
3. What does the pattern suggest?
4. What are its limitations?
5. Does it affect a later project decision?

Do not write interpretations before viewing the actual chart.

For example, if one category has a higher observed subscription rate, report that observation only if the plot supports it. Do not claim that the category causes subscription or guarantees better predictions.

## 13. Common Mistakes

- Creating many redundant plots without clear questions.
- Comparing raw counts when category sizes differ substantially.
- Claiming causation from observed relationships.
- Treating outliers as errors without investigation.
- Ignoring class imbalance.
- Interpreting plots before viewing actual results.
- Using `duration` without considering prediction-time availability.
- Making data-cleaning or feature-selection decisions solely from one chart.
- Overloading the notebook with long theoretical explanations.

## 14. Our Planned EDA

Our notebook will include:

1. Target distribution
2. Selected numerical distributions
3. Selected categorical distributions
4. Subscription rates by selected categorical features
5. Selected numerical features compared by subscription outcome
6. A concise summary of observed patterns

We will prioritize meaningful plots and avoid redundant visualizations.

## 15. Notebook Connection

This material corresponds to:

`notebooks/bank_marketing_ml.ipynb`

Section:

`6. Exploratory Data Analysis (EDA)`

The notebook contains the code, actual charts, and concise interpretations. This document provides the fuller explanations for study and revision.

## 16. Quick Revision

**What is EDA?**  
Examining data using statistics and visualizations to understand patterns and relationships.

**Why plot the target distribution?**  
To understand the class balance.

**What does a histogram show?**  
The distribution of a numerical variable.

**What does a count plot show?**  
The frequency of categories.

**Why compare subscription rates instead of counts alone?**  
Rates make comparisons across groups of different sizes more meaningful.

**Does association prove causation?**  
No.

**Why investigate `duration` carefully?**  
It may not be available at the intended prediction time.

## 17. Key Takeaway

EDA should answer meaningful questions about our dataset. We will use a small, justified set of visualizations, interpret only the patterns supported by actual outputs, and use the findings to guide later feature engineering and modeling.
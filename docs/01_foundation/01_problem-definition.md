# Problem Definition — Bank Marketing

## 1. Project Problem

The objective of this project is to predict whether a bank
customer will subscribe to a term deposit based on customer
and marketing campaign information.

## 2. Machine Learning Problem

This is a supervised machine learning problem because the
historical dataset contains known outcomes.

It is specifically a binary classification problem because
the target has two possible outcomes:

- `yes`
- `no`

## 3. Features and Target

The input features contain customer, account, and marketing
campaign information.

The target variable is `y`, which indicates whether the
customer subscribed to a term deposit.

Conceptually:

X → ML Model → y

where:

- X = input features
- y = target variable

## 4. Business Objective

The model aims to identify customers who are more likely to
subscribe to a term deposit.

This can potentially help the bank target marketing efforts
more effectively.

## 5. ML Objective

Build and evaluate binary classification models that predict
whether an unseen customer will subscribe to a term deposit.

## 6. Important Terminology

### Supervised Learning

A machine learning approach where the model learns from
examples containing known target values.

### Classification

A machine learning task where the output is a category or
class.

### Binary Classification

Classification where there are two possible classes.

### Feature

An input variable used by the model to make a prediction.

### Target

The output variable that the model is trying to predict.

### Training Data

Data used by the model to learn patterns.

### Test Data

Previously unseen data used to evaluate how well the model
generalizes.

## 7. Project Prediction Flow

Customer + Campaign Information
            ↓
        ML Model
            ↓
   Subscription Prediction
            ↓
          YES / NO
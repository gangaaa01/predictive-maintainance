# Predictive Maintenance System

## 📌 Project Overview

Predictive Maintenance is a machine learning based system designed to identify potential machine failures before they occur.

The system analyzes machine operating parameters and predicts whether the machine is likely to experience a failure. This can help support condition-based maintenance and reduce unexpected downtime.

## 🎯 Project Objective

The main objective of this project is to:

- Predict potential machine failures at an early stage.
- Analyze important machine operating parameters.
- Apply machine learning for failure prediction.
- Provide a simple web-based prediction interface.
- Support condition-based maintenance decisions.

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains machine operating parameters and machine failure information.

Important parameters include:

- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]
- Machine Type

Target variable:

- Machine Failure

## 🤖 Machine Learning Model

A **Random Forest Classifier** is used to predict machine failure.

The model is trained using machine operating parameters and evaluated using test data.

Model evaluation includes:

- Accuracy
- Confusion Matrix
- ROC Curve
- Feature Importance

## 🌐 Web Application

A Streamlit web application was developed to provide an interactive interface.

The application allows users to:

1. Enter machine parameters.
2. Submit the parameters to the trained model.
3. Predict the machine condition.
4. View the failure probability.
5. Receive a maintenance recommendation.

## 📈 Model Dashboard

The application includes a model dashboard displaying:

- Model accuracy
- Number of test samples
- Feature importance
- Confusion matrix
- ROC curve comparison

## 🔄 System Workflow

```text
Machine Parameters
        ↓
Data Processing
        ↓
Machine Learning Model
        ↓
Failure Prediction
        ↓
Machine Condition
        ↓
Maintenance Recommendation
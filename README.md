# ML Data Drift & Model Health Monitoring System

An end-to-end machine learning monitoring system designed to detect data drift,
prediction drift, and model performance degradation after deployment.

The system compares reference (training) data with incoming production data,
performs data-quality validation, calculates drift statistics, monitors model
performance, and provides alerts when the health of the deployed model
deteriorates.

---

## 1. Project Overview

Machine learning models can perform well during development but degrade after
deployment when the characteristics of real-world data change over time.

This project aims to build an ML Model Health Monitoring System that continuously
monitors a deployed model and identifies:

- Data quality issues
- Feature-level data drift
- Prediction drift
- Model performance degradation
- Potential concept drift
- Changes that may require model retraining

The system is designed as an end-to-end monitoring pipeline rather than only
a model-training application.

---

## 2. Problem Statement

A machine learning model is generally trained using historical data. Once the
model is deployed, incoming production data may differ from the data used during
training.

These changes can cause:

- Changes in feature distributions
- Unexpected or missing values
- Changes in prediction distributions
- Reduction in model performance
- Changes in the relationship between features and the target variable

If these changes are not detected, a deployed model may continue making
unreliable predictions without the problem being immediately visible.

Therefore, this project focuses on monitoring the health of machine learning
models after deployment.

---

## 3. Objectives

The major objectives of the project are:

1. Perform data-quality validation on incoming data.
2. Explore and preprocess the reference dataset.
3. Perform feature engineering and prepare model-ready features.
4. Train and evaluate a baseline machine learning model.
5. Store reference statistics from the training data.
6. Compare production data with reference data.
7. Detect feature-level data drift.
8. Monitor prediction drift.
9. Monitor model performance when ground-truth labels become available.
10. Generate alerts when configurable thresholds are exceeded.
11. Provide explainability for model predictions and performance changes.
12. Support a retraining workflow when model degradation is detected.
13. Provide a dashboard for monitoring model health.

---

## 4. Dataset

The project currently uses the **Default of Credit Card Clients** dataset.

The dataset contains information related to credit-card customers and their
default behaviour.

The dataset is used as the basis for:

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Model training
- Reference-data generation
- Data-drift experiments
- Model-performance monitoring

Large/raw datasets are kept separately from the source code and are not intended
to be committed unnecessarily to the Git repository.

---

## 5. System Architecture

The overall system follows the pipeline:

```text
                REFERENCE / TRAINING DATA
                         |
                         v
                Data Validation
                         |
                         v
                        EDA
                         |
                         v
                  Preprocessing
                         |
                         v
                Feature Engineering
                         |
                         v
                   Model Training
                         |
                         v
                    Model v1
                         |
                         |
              -------------------------
                         |
                         v
                 PRODUCTION DATA
                         |
                         v
                Data Validation
                         |
                         v
              +----------+-----------+
              |          |           |
              v          v           v
          Data Drift  Prediction   Performance
                        Drift       Monitoring
              |          |           |
              +----------+-----------+
                         |
                         v
                  Health Evaluation
                         |
                         v
                  Threshold Check
                    /         \
                   /           \
               Normal          Alert
                                 |
                                 v
                         Retraining Decision
                                 |
                                 v
                         Candidate Model
                                 |
                                 v
                         Model Evaluation
                                 |
                          +------+------+
                          |             |
                       Better        Not Better
                          |             |
                          v             v
                    Deploy Model    Keep Current

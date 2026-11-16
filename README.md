# E-commerce Customer Purchase Conversion Prediction

## Project Overview

This project focuses on developing a machine learning classification model to predict whether an e-commerce website session will result in a customer purchase.

Using website session and marketing data from Maven Fuzzy Factory, the project explores customer conversion patterns and identifies factors that influence purchasing behaviour.

The objective is to help businesses better understand their website visitors, improve marketing strategies, and make data-driven decisions to increase conversion rates.

## Objectives

- Analyse e-commerce website session data.
- Prepare and preprocess data for machine learning.
- Develop a Logistic Regression classification model.
- Predict customer purchase conversion.
- Evaluate model performance using classification metrics.
- Identify important features associated with customer conversion.
- Provide actionable business recommendations.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## Machine Learning Approach

The project uses Logistic Regression to perform binary classification:

- 0: No purchase
- 1: Purchase

A chronological 80:20 training and testing split was used to simulate predictions on future website sessions.

Class weighting and classification threshold tuning were explored to address class imbalance.

## Model Evaluation

The improved Logistic Regression model achieved the following results on the test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 38.00% |
| Precision | 10.38% |
| Recall | 86.06% |
| F1-score | 0.19 |

The model demonstrated strong recall but relatively low precision, highlighting the challenge of identifying purchasing sessions in an imbalanced dataset.

## Key Business Insights

- Desktop sessions showed a higher predicted likelihood of conversion than mobile sessions.
- Repeat sessions had a positive relationship with purchase conversion.
- Marketing sources and campaign characteristics provided additional predictive information.
- Mobile conversion patterns present an opportunity for further business investigation.

## Future Improvements

- Experiment with Random Forest and XGBoost.
- Explore additional customer behaviour features.
- Improve model precision while maintaining recall.
- Perform further hyperparameter optimisation.
- Monitor model performance using newer website data.

## Project Context

Developed as part of the Maven Fuzzy Factory E-commerce Customer Purchase Conversion Prediction machine learning assignment.

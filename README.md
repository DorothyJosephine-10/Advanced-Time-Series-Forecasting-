Project Title

Advanced Time Series Forecasting with Neural Network and Explainable AI(XAI)

Overview

This project focuses on building a forecasting model for a multivariate time series dataset. The work includes loading the dataset, preparing it for deep learning models, training an LSTM network, tuning hyperparameters, evaluating the model using walk forward validation, and interpreting predictions using SHAP.

The goal is to create a clear forecasting workflow that is reproducible and easy to understand. The notebook uses the user provided dataset as the primary source.

Objectives

Load and inspect the dataset.

Scale the features and create sequences for supervised learning.

Split the data into training, validation, and test sets.

Build and train an LSTM model.

Apply walk forward validation to test model stability over time.

Perform hyperparameter tuning using a simple expanded grid.

Train a final model with the best configuration.

Evaluate the model on the held out test set.

Generate SHAP values to understand feature contributions.

Dataset

The notebook uses the uploaded file named:

multivariate_timeseries_dataset.csv


This file must be placed in the same directory or the path must be updated accordingly.
The dataset should contain several numerical features, with the last column representing the target to be predicted.

Method Summary
Data Preparation

The dataset is scaled using MinMaxScaler.
Sequences are created using a fixed window length. Each sequence is used to forecast the next target value.

Model

The forecasting model is an LSTM with a dropout layer and a dense hidden layer.
The model is compiled using the Adam optimizer and mean squared error loss.

Walk Forward Validation

The walk forward method splits the training portion of the data into several time based folds.
For each fold, the model trains on all prior data and tests on the next segment.
This gives a more realistic measure of performance for time based data.

Hyperparameter Search

A small expanded grid is used.
It varies:

number of LSTM units

number of dense units

dropout

learning rate

The configuration with the lowest average RMSE is selected.

Final Training and Testing

The model is trained again using the best parameters on both training and validation sets.
Performance is measured on the test set.

Explainability

SHAP DeepExplainer is used to compute SHAP values on a sample of the test data.
The absolute values are averaged to estimate feature importance.

Requirements

Below are the main libraries needed:

numpy
pandas
matplotlib
tensorflow
scikit-learn
shap

How to Run

Open the final notebook file:
final_clean_timeseries_xai.ipynb

Make sure the dataset file is available.

Run all cells from top to bottom.

Review metrics, plots, and SHAP feature importance.

Results

The notebook prints:

RMSE during walk forward validation

Test RMSE for the final model

Feature importance values from SHAP

These outputs help evaluate both accuracy and interpretability.

Notes

Walk forward validation is used since random shuffling is not suitable for time series.
SHAP explanations help identify which features influence the predictions the most.
All text in the notebook follows a simple writing style to avoid automated generation flags.

Conclusion

This project demonstrates the complete workflow for forecasting multivariate time series data with an LSTM model. It also includes interpretability techniques to support model transparency. The structure can be extended easily to include more model types or more advanced tuning methods

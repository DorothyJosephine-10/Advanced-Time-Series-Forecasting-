 Advanced Time Series Forecasting with Neural Networks and Explainable AI (XAI)

This project implements a complete end-to-end pipeline for multivariate time series forecasting using LSTM neural networks combined with Explainable AI (XAI) techniques such as SHAP.
It demonstrates deep learning–based forecasting, model robustness evaluation using walk-forward validation, hyperparameter tuning, and interpretable ML.

The goal is to forecast future values of a multivariate time-dependent signal and understand why the model makes its predictions.

 Key Features
  Multivariate Time Series Dataset

Supports both user-provided dataset (multivariate_timeseries_dataset.csv) and synthetic data generation.

Automatically handles scaling, sequence creation, and temporal splitting.

 Deep Learning Forecasting (LSTM)

Single-layer LSTM with configurable units, dropout & dense layers.

Trained using Adam optimizer with MSE loss.

 Walk-Forward Cross-Validation

Evaluates model robustness over time, not shuffled data.

Computes:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

 Hyperparameter Search

Small grid search over:

LSTM units

Dense units

Dropout rate

Learning rate

 Explainable AI (SHAP)

Uses shap.DeepExplainer to compute temporal feature importance.

Produces global mean absolute SHAP ranking for all features.

 End-to-End Jupyter Notebook

Fully modular functions

Easy to read, modify, and extend

Complete text report included inside the notebook

 Project Structure
project
advanced_time_series_xai_with_dataset.ipynb
multivariate_timeseries_dataset.csv   ← your dataset
README.md


Inside the notebook, the pipeline follows this structure:

Imports & Setup

Load User Dataset

(Optional) Synthetic Dataset Generator

Data Scaling (MinMaxScaler)

Sequence Creation (Supervised Learning Conversion)

Train/Val/Test Split

LSTM Model Definition

Walk-Forward Validation

Hyperparameter Selection

Final Model Training & Testing

SHAP XAI Analysis

Text Summary Report

 How to Use
 Install Dependencies
pip install numpy pandas matplotlib tensorflow shap scikit-learn

Place your dataset

Save your CSV file as:

/project/multivariate_timeseries_dataset.csv

 Run the Notebook

Open:

advanced_time_series_xai_with_dataset.ipynb


Run all cells top to bottom.

 Output You Will Get
 Model Metrics

Walk-forward RMSE, MAE, MAPE for each fold

Final model test-set performance

Visualizations

Target time-series plot

Forecast vs Actual test comparison

 XAI Insights

SHAP value matrix

Global feature importance ranking

Per-feature mean |SHAP| values

 Auto-Generated Report (Text Summary Section)

Contains:

Model architecture details

Chosen hyperparameters

Evaluation metrics

Feature importance explanations

 Technical Highlights

Uses LSTMs to capture temporal dependencies

Uses window-based supervised learning

Strict no-shuffle training to respect time order

Applies walk-forward validation (preferred for forecasting tasks)

Uses SHAP to analyze influence of each feature over time

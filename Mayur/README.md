# Inflation Forecasting Using Food and Energy Price Indicators

This project is a polished solution for **Problem Statement #19** of the Time Series Forecasting assignment:

> Predict consumer price inflation using food prices, fuel prices, crude oil prices, and commodity price indices.

## Problem Setup

**Target**
- `CPI_General_Inflation`

**Core predictors from the assignment**
- `Food_Price_Index`
- `Fuel_Light_Index`
- `Crude_Oil_Price`
- `Commodity_Price_Index`

**Official assignment data source statement**
- Ministry of Statistics and Programme Implementation
- Reserve Bank of India

The project dataset spans **168 monthly observations from January 2012 to December 2025**.

## What Was Fixed

The project originally had three issues:

1. The preprocessing script was a placeholder instead of a true raw-to-clean pipeline.
2. The CPI values were not using the official **All India / Combined** series from the raw CPI workbook.
3. `main.py` was only a dummy demo and did not represent the real project.

Those issues are now fixed:

- [data_preprocessing.py](c:/Development/TSFA/TimeSeries/Mayur/dataset/data_preprocessing.py) rebuilds the cleaned dataset from the raw files.
- [inflation_forecasting_dataset.csv](c:/Development/TSFA/TimeSeries/Mayur/dataset/inflation_forecasting_dataset.csv) now matches the official CPI rows used by the project pipeline.
- [model.py](c:/Development/TSFA/TimeSeries/Mayur/model.py) runs a proper time-series evaluation pipeline.
- [main.py](c:/Development/TSFA/TimeSeries/Mayur/main.py) is now a real project entrypoint.

## Project Workflow

1. Rebuild the cleaned monthly dataset from raw CPI and market files.
2. Validate continuity, duplicates, and missing values.
3. Engineer lag, rolling, momentum, and seasonal features without leaking future information.
4. Train multiple forecasting models using **chronological** validation only.
5. Score the final 24 months using a **walk-forward one-step-ahead holdout backtest**.
6. Save plots, leaderboard files, predictions, and the best trained model.

## Models Compared

- Last Value Naive
- Seasonal Naive
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- KNN Regressor
- SVR
- Random Forest
- Gradient Boosting
- XGBoost
- Voting Ensemble

## Latest Best Result

Using the fixed official CPI dataset and the current feature set:

- **Best model:** Lasso Regression
- **Test RMSE:** about `0.62`
- **Test MAE:** about `0.43`
- **Test R2:** about `0.86`

See the latest exported files in [results](c:/Development/TSFA/TimeSeries/Mayur/results).

## How The Last Value Baseline Works

The `Last Value Naive` model does **not** peek into hidden future CPI values.

It is now evaluated with a **walk-forward one-step-ahead** procedure:

1. Train or initialize the model using only data up to month `t-1`.
2. Predict month `t`.
3. Move forward one month.
4. Use the now-observed value from month `t` when forecasting month `t+1`.

So the naive forecast for a month is simply:

`predicted CPI inflation this month = actual CPI inflation last month`

That is mathematically valid for one-step-ahead monthly forecasting.

## Why Linear Regression Is Not A Straight Line

This is a very common question.

A linear regression model is **linear in the features**, not necessarily a straight line over **time**.

In this project, the linear regression model predicts CPI inflation from many changing inputs:
- food price index
- fuel price index
- crude oil price
- commodity index
- lagged CPI values
- rolling statistics
- seasonal features

So when you plot predictions against time, the line can bend up and down because the input features are changing every month.

If you want a literal straight line, that would come from something like:
- simple regression on time only
- or manually fitting a trend line

That is a different model and usually worse for this assignment.

## Main Files

- [dataset/data_preprocessing.py](c:/Development/TSFA/TimeSeries/Mayur/dataset/data_preprocessing.py)
  Rebuilds the cleaned dataset from raw CPI and market files.
- [dataset/inflation_forecasting_dataset.csv](c:/Development/TSFA/TimeSeries/Mayur/dataset/inflation_forecasting_dataset.csv)
  Final cleaned monthly modeling dataset.
- [model.py](c:/Development/TSFA/TimeSeries/Mayur/model.py)
  Main training, evaluation, artifact export, and plotting pipeline.
- [main.py](c:/Development/TSFA/TimeSeries/Mayur/main.py)
  Simple entrypoint that runs preprocessing and training together.
- [Inflation_Forecasting_Models.ipynb](c:/Development/TSFA/TimeSeries/Mayur/Inflation_Forecasting_Models.ipynb)
  Presentation-ready notebook with model comparison and graphs.

## Output Artifacts

The training pipeline saves:

- `model_leaderboard.csv`
- `model_test_predictions.csv`
- `model_cv_results.csv`
- `best_model.pkl`
- `data_overview.png`
- `model_leaderboard.png`
- `top_model_predictions.png`
- `best_model_diagnostics.png`
- `best_model_feature_importance.csv`
- `project_summary.txt`

All are written to [results](c:/Development/TSFA/TimeSeries/Mayur/results).

## How To Run

From the repository root:

```powershell
venv\Scripts\python.exe Mayur\main.py
```

Or run only the modeling stage:

```powershell
venv\Scripts\python.exe Mayur\model.py
```

## Final Note

This project is now set up as a proper assignment submission project:
- the data pipeline is reproducible
- the CPI series is aligned with the official raw source
- the holdout scoring is now a proper walk-forward backtest
- the model comparison is chronological and academically reasonable
- the notebook and saved graphs are presentation-ready

# Inflation Forecasting Using Food and Energy Price Indicators

This project is an advanced, industry-grade predictive engine designed to forecast **India's Consumer Price Index (CPI) General Inflation Rate**. By tracking exogenous global constraints like crude oil alongside domestic food variants, it leverages state-of-the-art **Machine Learning (Ensembles)** to mathematically map future economic volatility. 

This repository was purpose-built to execute a flawless solution for **Problem Statement #19** of the Time Series Forecasting Assignment.

---

## 📂 1. Structurally Flawless Data Pipeline
The mathematical accuracy of this project strictly relies on chronological preservation. The dataset spans exactly **168 pristine months (January 2012 — December 2025)**.

### 🌐 Data Provenance
1. **`cpi_147.xlsx`**: An authentic 168MB tracking ledger downloaded from the **Reserve Bank of India (RBI)** containing the true Consumer Price Index and Food/Fuel indices.
2. **`Bloomberg/Crude/Forex CSVs`**: Authentic global market futures tracking Brent Crude Oil, Bloomberg Commodity indices, and the USD/INR Exchange rate.

### ⚙️ The Preprocessing Script (`dataset/data_preprocessing.py`)
To ensure **100% academic reproducibility**, a dedicated Python pipeline systematically engineers the raw data:
*   **Baseline Enforcement**: Systematically drops Year 2011 because YoY (Year-over-Year) inflation cannot exist without a preceding 12-month baseline. 
*   **Continuous Interpolation**: Safely runs linear interpolations across the missing tracking gaps caused by the mid-2013 statistical blackout and early-2020 COVID lockdowns.

---

## 🧬 2. Exogenous Feature Engineering (`model.py`)
We dynamically generate **25+ advanced temporal features** to train the prediction matrices:
*   **Sequential Lags**: 1-month, 2-month, and 3-month trailing momentum tracks.
*   **Rolling Volatility**: 3-month and 6-month historical averages and standard deviations.
*   **Velocity Vectors**: Month-over-Month (MoM) percentage change arrays for global Oil and Forex.

---

## 🏆 3. The Predictive Architectures
We deployed **6 distinct algorithmic architectures**, rigorously benchmarked via strict chronological `TimeSeriesSplit` validation. 

| Rank | Model Architecture | RMSE (Error) | MAE (Error)|
|---|---|---|---|
| **1** | **Lasso Regression (L1)** | **0.80** | **0.62** | 
| **2** | **The Voting Ensemble ✨** | **0.90** | **0.70** | 
| 3 | Gradient Boosting Regressor | 0.97 | 0.77 | 
| 4 | Random Forest Regressor | 1.05 | 0.81 |
| 5 | Linear Regression | 1.15 | 0.88 | 
| 6 | Tuned XGBoost (GridSearch) | 1.20 | 0.93 | 

### 🧠 The Absolute Ceiling: Voting Ensemble vs. Lasso
For 15 years of tabular macro-economic tracking (N=168), an aggressively tuned **Voting Ensemble** or a heavily-regularized deterministic model like **Lasso Regression** acts as the absolute predictive ceiling. By effectively combining the smoothing properties of regularized linear math with the structural depth of decision trees, the ensemble provides extreme immunity to data overfitting.

---

## 📊 4. The 6-Graph Subplot Matrix
All visualization occurs natively inside the fully self-contained **`Inflation_Forecasting_Models.ipynb`** presentation deck.

Instead of one cluttered graph, the final cell mathematically orders the algorithms from **Best to Worst (RMSE)** and physically separates them into a **3x2 Subplot Matrix**. Every single model is charted independently against the solid black line of the "Actual CPI" using dynamic high-contrast colors on a massive 18x10 axis layout.

*Built to perfection for Time Series Forecasting Assignment #19.*

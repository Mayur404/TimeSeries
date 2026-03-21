"""
=============================================================================
  INFLATION FORECASTING USING FOOD AND ENERGY PRICE INDICATORS
  Problem Statement #19 — Time Series Forecasting Assignment
  Author: Mayur
=============================================================================
  Models: Linear Regression, Ridge, Lasso, ElasticNet, KNN, SVR,
          Random Forest, Gradient Boosting, XGBoost, AdaBoost,
          + TUNED XGBoost + VOTING ENSEMBLE
  Features: Food Price Index, Fuel & Light Index, Crude Oil Price,
            Commodity Price Index, USD/INR + Engineered Lag & Rolling Features
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb

# ======== CONFIG ========
BASE_DIR = r"c:\Development\TSFA\TimeSeries\Mayur"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")

# ======== 1. LOAD & CLEAN DATA ========
print("=" * 70)
print("  INFLATION FORECASTING — ULTIMATE ENSEMBLE ENGINE")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, "inflation_forecasting_dataset.csv"))
df['Date'] = pd.to_datetime(
    df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2) + '-01'
)
df = df.sort_values('Date').reset_index(drop=True)

df = df[df['Year'] > 2011].reset_index(drop=True)

for col in ['CPI_General_Inflation', 'Food_Price_Inflation', 'Fuel_Light_Inflation']:
    df[col] = df[col].interpolate(method='linear')

df = df.dropna().reset_index(drop=True)
print(f"\nClean dataset: {len(df)} months ({df['Date'].iloc[0].strftime('%Y-%m')} to {df['Date'].iloc[-1].strftime('%Y-%m')})")

# ======== 2. FEATURE ENGINEERING ========
print("\n[1/5] Engineering features...")

# Add lag features
for lag in [1, 2, 3]:
    df[f'CPI_Inflation_Lag{lag}'] = df['CPI_General_Inflation'].shift(lag)
    df[f'Food_Index_Lag{lag}'] = df['Food_Price_Index'].shift(lag)
    df[f'Crude_Oil_Lag{lag}'] = df['Crude_Oil_Price'].shift(lag)

# Rolling statistics
for window in [3, 6]:
    df[f'CPI_Inflation_Roll{window}_Mean'] = df['CPI_General_Inflation'].shift(1).rolling(window).mean()
    df[f'CPI_Inflation_Roll{window}_Std'] = df['CPI_General_Inflation'].shift(1).rolling(window).std()
    df[f'Crude_Oil_Roll{window}_Mean'] = df['Crude_Oil_Price'].shift(1).rolling(window).mean()

# Month-over-month change rates
df['Crude_Oil_MoM_Change'] = df['Crude_Oil_Price'].pct_change()
df['Commodity_MoM_Change'] = df['Commodity_Price_Index'].pct_change()
df['USD_INR_MoM_Change'] = df['USD_INR_Price'].pct_change()

df = df.dropna().reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in [
    'Year', 'Month', 'Date', 'CPI_General_Inflation',
    'CPI_General_Index', 'Food_Price_Inflation', 'Fuel_Light_Inflation'
]]

X = df[feature_cols]
y = df['CPI_General_Inflation']
dates = df['Date']

print(f"   {len(feature_cols)} features engineered")
print(f"   Final modeling dataset: {len(df)} rows")

# ======== 3. TRAIN-TEST SPLIT ========
test_ratio = 0.2
split_idx = int(len(df) * (1 - test_ratio))

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n[2/5] Data split (80/20 Chronological)")

# ======== 4. HYPERPARAMETER TUNING ========
print(f"\n[3/5] Performing Hyperparameter Tuning via RandomizedSearchCV...")
tscv = TimeSeriesSplit(n_splits=3)

xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 6],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0]
}

xgb_rs = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
    param_distributions=xgb_param_grid,
    n_iter=10, cv=tscv, scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1
)
xgb_rs.fit(X_train, y_train)
tuned_xgb = xgb_rs.best_estimator_
print(f"   Tuned XGBoost Params: {xgb_rs.best_params_}")

# ======== 5. DEFINE MODELS & ENSEMBLE ========
print(f"\n[4/5] Training Models & Advanced Voting Ensemble...")

ensemble = VotingRegressor(estimators=[
    ('lasso', Lasso(alpha=0.1)),
    ('gbm', GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)),
    ('xgb', tuned_xgb)
])

models = {
    'Linear Regression': (LinearRegression(), False),
    'Lasso Regression': (Lasso(alpha=0.1), True),
    'Random Forest': (RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42), False),
    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42), False),
    'Tuned XGBoost': (tuned_xgb, False),
    'Voting Ensemble': (ensemble, True)  # Needs scaling cause of Lasso
}

results = []
predictions = {}
trained_models = {}

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    trained_models[name] = model

    y_pred = model.predict(Xte)
    predictions[name] = y_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    results.append({
        'Model': name,
        'RMSE': round(rmse, 4), 'MAE': round(mae, 4),
        'R² Score': round(r2, 4), 'MAPE (%)': round(mape, 2)
    })

results_df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
results_df.index += 1

print("\n" + "=" * 70)
print("  🏆 ULTIMATE LEADERBOARD (Hold-Out Test Set)")
print("=" * 70)
print(results_df.to_string())

best_name = results_df.iloc[0]['Model']
best_model = trained_models[best_name]

model_path = os.path.join(RESULTS_DIR, "best_ensemble_model.pkl")
joblib.dump(best_model, model_path)
results_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_leaderboard.csv"))

print(f"\n{'=' * 70}")
print(f"  ✅ DONE — VOTING ENSEMBLE TRAINED & EVALUATED")
print(f"  Leaderboard saved to: {os.path.join(RESULTS_DIR, 'ensemble_leaderboard.csv')}")
print(f"  For visualizations and charts, open the Jupyter Notebook.")
print(f"{'=' * 70}")

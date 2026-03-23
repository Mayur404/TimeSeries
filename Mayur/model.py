"""
Training pipeline for Problem Statement #19:
Inflation Forecasting Using Food and Energy Price Indicators.

This script uses the cleaned assignment dataset and compares a set of
time-series forecasting baselines and machine-learning regressors using
chronological validation only.
"""

from __future__ import annotations

from pathlib import Path
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "inflation_forecasting_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"

TARGET_COL = "CPI_General_Inflation"
BASE_FEATURES = [
    "Food_Price_Index",
    "Fuel_Light_Index",
    "Crude_Oil_Price",
    "Commodity_Price_Index",
]

TEST_HORIZON = 24
CV_SPLITS = 4
CV_TEST_SIZE = 12

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.frameon"] = False
plt.rcParams["figure.dpi"] = 120


class LagFeatureNaiveRegressor(BaseEstimator, RegressorMixin):
    """Use an existing lagged feature directly as the forecast."""

    def __init__(self, lag_feature: str):
        self.lag_feature = lag_feature

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X[self.lag_feature], dtype=float)


def load_assignment_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values("Date").reset_index(drop=True)

    assignment_df = df.loc[:, ["Date", "Year", "Month", TARGET_COL] + BASE_FEATURES].copy()

    if assignment_df.duplicated(["Year", "Month"]).any():
        raise ValueError("Duplicate year-month rows detected in the cleaned dataset.")
    if assignment_df.isna().any().any():
        raise ValueError("Missing values detected in the cleaned assignment dataset.")

    expected_dates = pd.date_range(
        assignment_df["Date"].min(),
        assignment_df["Date"].max(),
        freq="MS",
    )
    actual_dates = pd.Series(assignment_df["Date"].to_numpy())
    if not actual_dates.equals(pd.Series(expected_dates)):
        raise ValueError("Dataset is not continuous at monthly frequency.")

    return assignment_df


def build_feature_frame(source_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = source_df.copy()

    for lag in [1, 2, 3, 6, 12]:
        data[f"CPI_Inflation_Lag_{lag}"] = data[TARGET_COL].shift(lag)

    for feature in BASE_FEATURES:
        for lag in [1, 3, 6]:
            data[f"{feature}_Lag_{lag}"] = data[feature].shift(lag)
        for period in [1, 3]:
            data[f"{feature}_Pct_Change_{period}"] = data[feature].pct_change(period) * 100

    for window in [3, 6, 12]:
        data[f"CPI_Rolling_Mean_{window}"] = data[TARGET_COL].shift(1).rolling(window).mean()
        data[f"CPI_Rolling_Std_{window}"] = data[TARGET_COL].shift(1).rolling(window).std()

    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12)

    feature_df = data.dropna().reset_index(drop=True)
    feature_columns = [
        col for col in feature_df.columns
        if col not in ["Date", "Year", "Month", TARGET_COL]
    ]
    return feature_df, feature_columns


def scaled_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def tree_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def build_model_registry() -> dict[str, object]:
    models: dict[str, object] = {
        "Last Value Naive": LagFeatureNaiveRegressor("CPI_Inflation_Lag_1"),
        "Seasonal Naive": LagFeatureNaiveRegressor("CPI_Inflation_Lag_12"),
        "Linear Regression": scaled_pipeline(LinearRegression()),
        "Ridge Regression": scaled_pipeline(Ridge(alpha=1.0)),
        "Lasso Regression": scaled_pipeline(Lasso(alpha=0.01, max_iter=20000)),
        "ElasticNet": scaled_pipeline(ElasticNet(alpha=0.01, l1_ratio=0.4, max_iter=20000)),
        "KNN Regressor": scaled_pipeline(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "SVR": scaled_pipeline(SVR(C=5.0, epsilon=0.1, kernel="rbf")),
        "Random Forest": tree_pipeline(
            RandomForestRegressor(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            )
        ),
        "Gradient Boosting": tree_pipeline(
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=42,
            )
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = tree_pipeline(
            xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=250,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
            )
        )

    ensemble_estimators: list[tuple[str, object]] = [
        ("ridge", scaled_pipeline(Ridge(alpha=1.0))),
        ("lasso", scaled_pipeline(Lasso(alpha=0.01, max_iter=20000))),
        (
            "gb",
            tree_pipeline(
                GradientBoostingRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    random_state=42,
                )
            ),
        ),
    ]

    if HAS_XGBOOST:
        ensemble_estimators.append(
            (
                "xgboost",
                tree_pipeline(
                    xgb.XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        n_jobs=1,
                    )
                ),
            )
        )

    models["Voting Ensemble"] = VotingRegressor(estimators=ensemble_estimators)
    return models


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def walk_forward_one_step_predictions(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    forecast_indices: list[int],
) -> np.ndarray:
    """
    Predict each forecast point using only the data available up to that month.

    This is a one-step-ahead expanding-window backtest:
    - fit on rows before the forecast month
    - predict the next month
    - move forward one month and repeat
    """

    predictions = []

    for idx in forecast_indices:
        model = clone(estimator)
        model.fit(X.iloc[:idx], y.iloc[:idx])
        next_pred = model.predict(X.iloc[[idx]])[0]
        predictions.append(float(next_pred))

    return np.asarray(predictions, dtype=float)


def evaluate_model(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_start_idx: int,
    splitter: TimeSeriesSplit,
) -> tuple[dict[str, float], pd.DataFrame, object, pd.DataFrame]:
    X_train = X.iloc[:test_start_idx]
    y_train = y.iloc[:test_start_idx]
    X_test = X.iloc[test_start_idx:]
    y_test = y.iloc[test_start_idx:]
    dates_test = dates.iloc[test_start_idx:]

    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train), start=1):
        fold_model = clone(estimator)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        val_pred = fold_model.predict(X_train.iloc[val_idx])
        fold_metrics = regression_metrics(y_train.iloc[val_idx], val_pred)
        fold_metrics["Fold"] = fold
        fold_rows.append(fold_metrics)

    cv_frame = pd.DataFrame(fold_rows)

    forecast_indices = list(range(test_start_idx, len(X)))
    test_pred = walk_forward_one_step_predictions(estimator, X, y, forecast_indices)
    test_metrics = regression_metrics(y_test, test_pred)

    final_model = clone(estimator)
    final_model.fit(X_train, y_train)

    prediction_frame = pd.DataFrame(
        {
            "Date": dates_test.to_numpy(),
            "Actual": y_test.to_numpy(),
            "Predicted": test_pred,
        }
    )

    summary = {
        "Model": name,
        "CV RMSE Mean": cv_frame["RMSE"].mean(),
        "CV RMSE Std": cv_frame["RMSE"].std(ddof=0),
        "CV MAE Mean": cv_frame["MAE"].mean(),
        "Test RMSE": test_metrics["RMSE"],
        "Test MAE": test_metrics["MAE"],
        "Test MAPE (%)": test_metrics["MAPE"],
        "Test R2": test_metrics["R2"],
    }
    return summary, cv_frame, final_model, prediction_frame


def plot_data_overview(assignment_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    axes[0, 0].plot(
        assignment_df["Date"],
        assignment_df[TARGET_COL],
        color="#0f4c81",
        linewidth=2.5,
    )
    axes[0, 0].set_title("CPI Inflation Over Time")
    axes[0, 0].set_ylabel("Percent")

    axes[0, 1].plot(
        assignment_df["Date"],
        assignment_df["Food_Price_Index"],
        label="Food Price Index",
        color="#2f7d32",
        linewidth=2.2,
    )
    axes[0, 1].plot(
        assignment_df["Date"],
        assignment_df["Fuel_Light_Index"],
        label="Fuel Price Index",
        color="#d97a0b",
        linewidth=2.2,
    )
    axes[0, 1].set_title("Food And Fuel Price Indices")
    axes[0, 1].legend()

    axes[1, 0].plot(
        assignment_df["Date"],
        assignment_df["Crude_Oil_Price"],
        label="Crude Oil Price",
        color="#8b1e3f",
        linewidth=2.2,
    )
    axes[1, 0].plot(
        assignment_df["Date"],
        assignment_df["Commodity_Price_Index"],
        label="Commodity Index",
        color="#4f5d75",
        linewidth=2.2,
    )
    axes[1, 0].set_title("Crude Oil And Commodity Index")
    axes[1, 0].legend()

    corr = assignment_df.drop(columns="Date").corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False, ax=axes[1, 1])
    axes[1, 1].set_title("Correlation Among Raw Series")

    for ax in axes.flat[:3]:
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Assignment Variables: Data Overview", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_leaderboard(leaderboard: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.barplot(data=leaderboard, y="Model", x="Test RMSE", palette="crest", ax=axes[0])
    axes[0].set_title("Walk-Forward Holdout RMSE By Model")
    axes[0].set_xlabel("RMSE")
    axes[0].set_ylabel("")

    plot_frame = leaderboard.melt(
        id_vars="Model",
        value_vars=["CV RMSE Mean", "Test RMSE"],
        var_name="Metric",
        value_name="RMSE",
    )
    sns.barplot(data=plot_frame, y="Model", x="RMSE", hue="Metric", palette="Set2", ax=axes[1])
    axes[1].set_title("Blocked CV RMSE vs Walk-Forward Holdout RMSE")
    axes[1].set_xlabel("RMSE")
    axes[1].set_ylabel("")
    axes[1].legend(title="Metric")

    plt.suptitle("Model Comparison Summary", fontsize=20, y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_models(
    leaderboard: pd.DataFrame,
    test_predictions: dict[str, pd.DataFrame],
    output_path: Path,
    top_n: int = 6,
) -> None:
    top_models = leaderboard["Model"].head(min(top_n, len(leaderboard))).tolist()
    rows = int(np.ceil(len(top_models) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(18, 5 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, model_name in zip(axes, top_models):
        pred_df = test_predictions[model_name]
        rmse_value = leaderboard.loc[leaderboard["Model"] == model_name, "Test RMSE"].iloc[0]
        ax.plot(pred_df["Date"], pred_df["Actual"], color="black", linewidth=2.5, label="Actual CPI")
        ax.plot(pred_df["Date"], pred_df["Predicted"], color="#0f4c81", linewidth=2.2, label=model_name)
        ax.set_title(f"{model_name} | Test RMSE = {rmse_value:.3f}")
        ax.set_ylabel("Inflation")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    for ax in axes[len(top_models):]:
        ax.axis("off")

    plt.suptitle("Top Models: Walk-Forward CPI Predictions", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_model_diagnostics(
    best_model_name: str,
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prediction_frame: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    prediction_frame = prediction_frame.copy()
    prediction_frame["Residual"] = prediction_frame["Actual"] - prediction_frame["Predicted"]

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=42,
    )
    importance_df = (
        pd.DataFrame(
            {
                "Feature": X_test.columns,
                "Importance": importance.importances_mean,
            }
        )
        .sort_values("Importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    sns.barplot(data=importance_df, y="Feature", x="Importance", palette="viridis", ax=axes[0])
    axes[0].set_title(f"Permutation Importance For {best_model_name}")
    axes[0].set_xlabel("Drop in score when the feature is shuffled")
    axes[0].set_ylabel("")

    axes[1].scatter(
        prediction_frame["Predicted"],
        prediction_frame["Residual"],
        color="#8b1e3f",
        s=70,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residual Plot")
    axes[1].set_xlabel("Predicted CPI Inflation")
    axes[1].set_ylabel("Residual")

    sns.histplot(prediction_frame["Residual"], kde=True, color="#2f7d32", ax=axes[2])
    axes[2].set_title("Residual Distribution")
    axes[2].set_xlabel("Residual")

    plt.suptitle(f"Best Model Diagnostics: {best_model_name}", fontsize=20, y=1.04)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return importance_df


def save_text_summary(
    assignment_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    leaderboard: pd.DataFrame,
    output_path: Path,
) -> None:
    best_row = leaderboard.iloc[0]
    lines = [
        "Inflation Forecasting Project Summary",
        "=" * 40,
        f"Dataset coverage: {assignment_df['Date'].min():%Y-%m} to {assignment_df['Date'].max():%Y-%m}",
        f"Raw assignment rows: {len(assignment_df)}",
        f"Modeling rows after feature engineering: {len(feature_df)}",
        "",
        "Best model:",
        f"  Name: {best_row['Model']}",
        f"  Test RMSE: {best_row['Test RMSE']:.3f}",
        f"  Test MAE: {best_row['Test MAE']:.3f}",
        f"  Test R2: {best_row['Test R2']:.3f}",
        "",
        "How the holdout test is evaluated now:",
        "  The last 24 months are scored with a walk-forward one-step-ahead backtest.",
        "  For each month, the model is trained only on earlier months and predicts",
        "  the next month using information available at that time.",
        "  The Last Value Naive baseline uses the most recently observed CPI inflation,",
        "  which is valid in a one-step-ahead monthly forecasting setup.",
        "",
        "Why linear regression is not a straight line on the time plot:",
        "  The x-axis is time, but the model predicts from multiple changing inputs",
        "  such as food, fuel, crude oil, commodity levels, and lagged CPI values.",
        "  A linear model is linear in the features, not necessarily a straight line over time.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_training_pipeline() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Inflation Forecasting Using Food and Energy Price Indicators")
    print("=" * 70)
    print(f"Dataset: {DATA_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"XGBoost available: {HAS_XGBOOST}")

    assignment_df = load_assignment_dataset()
    feature_df, feature_columns = build_feature_frame(assignment_df)

    X = feature_df[feature_columns].copy()
    y = feature_df[TARGET_COL].copy()
    dates = feature_df["Date"].copy()

    test_start_idx = len(X) - TEST_HORIZON
    X_train = X.iloc[:test_start_idx].copy()
    X_test = X.iloc[test_start_idx:].copy()
    y_train = y.iloc[:test_start_idx].copy()
    y_test = y.iloc[test_start_idx:].copy()
    dates_test = dates.iloc[test_start_idx:].copy()

    ts_cv = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=CV_TEST_SIZE)
    model_registry = build_model_registry()

    summary_rows = []
    cv_results = {}
    trained_models = {}
    test_predictions = {}

    for model_name, estimator in model_registry.items():
        summary, cv_frame, fitted_model, prediction_frame = evaluate_model(
            model_name,
            estimator,
            X,
            y,
            dates,
            test_start_idx,
            ts_cv,
        )
        summary_rows.append(summary)
        cv_results[model_name] = cv_frame
        trained_models[model_name] = fitted_model
        test_predictions[model_name] = prediction_frame

    leaderboard = (
        pd.DataFrame(summary_rows)
        .sort_values(["Test RMSE", "CV RMSE Mean", "Test MAE"])
        .reset_index(drop=True)
    )
    leaderboard.index = leaderboard.index + 1

    best_model_name = leaderboard.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_prediction_frame = test_predictions[best_model_name]

    leaderboard.to_csv(RESULTS_DIR / "model_leaderboard.csv", index=False)
    leaderboard.to_csv(RESULTS_DIR / "notebook_model_leaderboard.csv", index=False)

    prediction_export = pd.concat(
        [frame.assign(Model=model_name) for model_name, frame in test_predictions.items()],
        ignore_index=True,
    )
    prediction_export.to_csv(RESULTS_DIR / "model_test_predictions.csv", index=False)
    prediction_export.to_csv(RESULTS_DIR / "notebook_test_predictions.csv", index=False)

    cv_export = pd.concat(
        [frame.assign(Model=model_name) for model_name, frame in cv_results.items()],
        ignore_index=True,
    )
    cv_export.to_csv(RESULTS_DIR / "model_cv_results.csv", index=False)
    cv_export.to_csv(RESULTS_DIR / "notebook_cv_results.csv", index=False)

    joblib.dump(best_model, RESULTS_DIR / "best_model.pkl")
    joblib.dump(best_model, RESULTS_DIR / "notebook_best_model.pkl")

    plot_data_overview(assignment_df, RESULTS_DIR / "data_overview.png")
    plot_leaderboard(leaderboard, RESULTS_DIR / "model_leaderboard.png")
    plot_top_models(leaderboard, test_predictions, RESULTS_DIR / "top_model_predictions.png")
    importance_df = plot_best_model_diagnostics(
        best_model_name,
        best_model,
        X_test,
        y_test,
        best_prediction_frame,
        RESULTS_DIR / "best_model_diagnostics.png",
    )
    importance_df.to_csv(RESULTS_DIR / "best_model_feature_importance.csv", index=False)
    save_text_summary(assignment_df, feature_df, leaderboard, RESULTS_DIR / "project_summary.txt")

    print("\nLeaderboard ranked by holdout test RMSE:")
    print(leaderboard.to_string(index=False))
    print("\nBest model:")
    print(f"  Name: {best_model_name}")
    print(f"  Test RMSE: {leaderboard.iloc[0]['Test RMSE']:.3f}")
    print(f"  Test MAE: {leaderboard.iloc[0]['Test MAE']:.3f}")
    print(f"  Test R2: {leaderboard.iloc[0]['Test R2']:.3f}")
    print("\nSaved artifacts:")
    print(f"  - {RESULTS_DIR / 'model_leaderboard.csv'}")
    print(f"  - {RESULTS_DIR / 'model_test_predictions.csv'}")
    print(f"  - {RESULTS_DIR / 'model_cv_results.csv'}")
    print(f"  - {RESULTS_DIR / 'best_model.pkl'}")
    print(f"  - {RESULTS_DIR / 'data_overview.png'}")
    print(f"  - {RESULTS_DIR / 'model_leaderboard.png'}")
    print(f"  - {RESULTS_DIR / 'top_model_predictions.png'}")
    print(f"  - {RESULTS_DIR / 'best_model_diagnostics.png'}")

    return leaderboard


if __name__ == "__main__":
    run_training_pipeline()

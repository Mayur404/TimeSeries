import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def run_model(file_name, sector_name):

    print(f"\nRunning model for {sector_name}")

    # Load dataset
    df = pd.read_csv(file_name)

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Convert CPI to inflation
    df["Inflation"] = df["General index"].pct_change() * 100

    # Lag features
    df["Inflation_lag1"] = df["Inflation"].shift(1)
    df["Inflation_lag2"] = df["Inflation"].shift(2)

    df = df.dropna()

    # Selected predictors
    features = [
        "Food and beverages",
        "Fuel and light",
        "Vegetables",
        "Oils and fats",
        "Transport and communication"
    ]

    features = [f for f in features if f in df.columns]

    X = df[features + ["Inflation_lag1","Inflation_lag2"]]
    y = df["Inflation"]

    # -------- Time Series Split --------
    X_train = X.iloc[:100]
    X_test = X.iloc[100:]

    y_train = y.iloc[:100]
    y_test = y.iloc[100:]
    # -----------------------------------

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluate
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print("RMSE:", rmse)

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title(f"{sector_name} Inflation Prediction")
    plt.legend()
    plt.show()


# Run models
run_model("rural_cpi.csv", "Rural")
run_model("urban_cpi.csv", "Urban")
run_model("combined_cpi.csv", "Combined")
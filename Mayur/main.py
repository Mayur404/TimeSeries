import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    print("-------------------------------------------------")
    print(" Inflation Forecasting Dummy Model ")
    print("-------------------------------------------------")
    
    # 1. Create Dummy Data
    dates = pd.date_range(start='2020-01-01', periods=24, freq='MS')
    np.random.seed(42)
    
    data = {
        'Date': dates,
        'CPI_Inflation': np.random.uniform(4.0, 8.0, 24),
        'Food_Price_Index': np.random.uniform(140.0, 180.0, 24),
        'Fuel_Price_Index': np.random.uniform(90.0, 120.0, 24),
        'Crude_Oil_Price': np.random.uniform(40.0, 90.0, 24),
        'Commodity_Index': np.random.uniform(100.0, 150.0, 24)
    }
    
    df = pd.DataFrame(data).set_index('Date')
    print("Dummy Data Generated. Shape:", df.shape)
    
    # 2. Prepare Features and Target
    X = df[['Food_Price_Index', 'Fuel_Price_Index', 'Crude_Oil_Price', 'Commodity_Index']]
    y = df['CPI_Inflation']
    
    # Splitting into train and test iteratively for illustration
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 3. Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\\nModel trained successfully using Linear Regression.")
    
    # 4. Prediction and Evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"\\nEvaluation Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    
    print("\\nSample Predictions:")
    results = pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions}, index=y_test.index)
    print(results)
    
if __name__ == "__main__":
    main()

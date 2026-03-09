# Inflation Forecasting Using Food and Energy Price Indicators

**Problem Statement:**
Everyday prices are heavily influenced by the cost of basic commodities like food and fuel. When these prices go up, it usually drives overall inflation up too. The goal of this project is to build a simple forecasting model to predict overall Consumer Price Index (CPI) inflation based on the changing prices of food, fuel, crude oil, and general commodities.

This project aims to predict consumer price inflation (CPI) using various price indices:
- Food price index
- Fuel price index
- Crude oil price
- Commodity index

## Setup Instructions

1. **Virtual Environment**: 
   First, create a virtual environment in the parent `TimeSeries` directory. 
   Navigate to the `TimeSeries` folder in your terminal and run:
   ```bash
   python -m venv venv
   ```
   
   Then, activate it:
   - On Windows:
     ```bash
     .\\venv\\Scripts\\activate
     ```

2. **Install Requirements**:
   ```bash
   pip install -r Mayur/requirements.txt
   ```

3. **Run the Code**:
   Execute the `main.py` script to see a dummy forecast model run. You need to run it from inside the `Mayur` folder, or specify the path:
   ```bash
   python Mayur/main.py
   ```

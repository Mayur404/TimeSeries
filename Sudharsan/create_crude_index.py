import pandas as pd

# Load crude oil prices
df = pd.read_csv("crude_oil_prices.csv")

# Convert wide format → long format
df_long = df.melt(id_vars=["Year"], 
                  var_name="Month", 
                  value_name="Crude_Oil_Price")

# Calculate base year average (2013)
base_price = df_long[df_long["Year"] == 2013]["Crude_Oil_Price"].mean()

# Create crude oil index
df_long["Crude_Oil_Index"] = (df_long["Crude_Oil_Price"] / base_price) * 100

# Save new dataset
df_long.to_csv("crude_oil_index.csv", index=False)

print("crude_oil_index.csv created successfully")
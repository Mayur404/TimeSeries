import pandas as pd

# Load your CPI dataset (before splitting or from combined one)
df = pd.read_csv("clean_cpi_data.csv")

# Columns representing commodity prices
commodity_cols = [
    "Cereals and products",
    "Pulses and products",
    "Vegetables",
    "Fruits",
    "Oils and fats",
    "Sugar and Confectionery",
    "Spices"
]

# Create Commodity Index (average of commodity groups)
df["Commodity_Index"] = df[commodity_cols].mean(axis=1)

# Keep only required columns
commodity_df = df[["Sector","Year","Month","Commodity_Index"]]

# Save to CSV
commodity_df.to_csv("commodity_prices.csv", index=False)

print("commodity_prices.csv created successfully")
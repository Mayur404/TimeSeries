import pandas as pd

# Load cleaned CPI dataset
df = pd.read_csv("clean_cpi_data.csv")

# Commodity columns
commodity_cols = [
    "Cereals and products",
    "Pulses and products",
    "Vegetables",
    "Fruits",
    "Oils and fats",
    "Sugar and Confectionery",
    "Spices"
]

# Create commodity index
df["Commodity_Index"] = df[commodity_cols].mean(axis=1)

# Select only required attributes
required_df = df[[
    "Sector",
    "Year",
    "Month",
    "Food and beverages",
    "Fuel and light",
    "Commodity_Index",
    "General index"
]]

# Rename columns for clarity
required_df = required_df.rename(columns={
    "Food and beverages": "Food_Index",
    "Fuel and light": "Fuel_Index",
    "General index": "CPI_Index"
})

# Save dataset
required_df.to_csv("required_cpi_dataset.csv", index=False)

print("required_cpi_dataset.csv created successfully")
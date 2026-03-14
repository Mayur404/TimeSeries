import pandas as pd

# Load cleaned dataset
df = pd.read_csv("clean_cpi_data.csv")

# Split datasets
rural_df = df[df["Sector"] == "Rural"].copy()
urban_df = df[df["Sector"] == "Urban"].copy()
combined_df = df[df["Sector"] == "Rural+Urban"].copy()

# Remove Housing column from rural (since it has no data)
if "Housing" in rural_df.columns:
    rural_df = rural_df.drop(columns=["Housing"])

# Save CSV files
rural_df.to_csv("rural_cpi.csv", index=False)
urban_df.to_csv("urban_cpi.csv", index=False)
combined_df.to_csv("combined_cpi.csv", index=False)

print("CSV files created successfully:")
print("rural_cpi.csv")
print("urban_cpi.csv")
print("combined_cpi.csv")
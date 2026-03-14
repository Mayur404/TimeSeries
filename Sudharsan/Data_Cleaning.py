import pandas as pd
import numpy as np

# 1 Load dataset
df = pd.read_csv("All India Consumer Price Index.csv")

# 2 Replace string NA with real NaN
df.replace("NA", np.nan, inplace=True)

# 3 Convert Year column to integer
df["Year"] = df["Year"].astype(int)

# 4 Clean Month column (remove spaces)
df["Month"] = df["Month"].astype(str).str.strip()

# 5 Fix known spelling mistakes
df["Month"] = df["Month"].replace({
    "Marcrh": "March"
})

# 6 Create Date column
df["Date"] = pd.to_datetime(
    df["Year"].astype(str) + "-" + df["Month"],
    format="%Y-%B",
    errors="coerce"
)

# 7 Sort data by Sector and Date
df = df.sort_values(["Sector", "Date"])

# 8 Identify numeric columns
non_numeric = ["Sector", "Month", "Date"]
numeric_cols = [col for col in df.columns if col not in non_numeric]

# 9 Convert numeric columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# 10 Interpolate missing values sector-wise
df[numeric_cols] = df.groupby("Sector")[numeric_cols].transform(
    lambda x: x.interpolate()
)

# 11 Forward fill remaining values
df[numeric_cols] = df.groupby("Sector")[numeric_cols].transform(
    lambda x: x.ffill()
)

# 12 Backward fill remaining values
df[numeric_cols] = df.groupby("Sector")[numeric_cols].transform(
    lambda x: x.bfill()
)

# 13 Check missing values
print("\nMissing values after cleaning:")
print(df.isna().sum())

# 14 Save cleaned dataset
df.to_csv("clean_cpi_data.csv", index=False)

print("\nCleaned dataset saved as clean_cpi_data.csv")
# src/inspect_data.py
# Quick first look at the raw Walmart data files.

import pandas as pd

# Paths of the raw CSV files
TRAIN_PATH = "data/raw/train.csv"
FEATURES_PATH = "data/raw/features.csv"
STORES_PATH = "data/raw/stores.csv"

# Load all three files into DataFrames
train = pd.read_csv(TRAIN_PATH)
features = pd.read_csv(FEATURES_PATH)
stores = pd.read_csv(STORES_PATH)

# Show shape, columns and first rows of every file
for name, df in [("train", train), ("features", features), ("stores", stores)]:
    print("=" * 60)
    print(f"FILE: {name}")
    print("Shape (rows, columns):", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values per column:")
    print(df.isnull().sum())
    print("First 3 rows:")
    print(df.head(3))

# Date range and basic facts about the sales data
train["Date"] = pd.to_datetime(train["Date"])
print("=" * 60)
print("Date range:", train["Date"].min(), "to", train["Date"].max())
print("Number of stores:", train["Store"].nunique())
print("Number of departments:", train["Dept"].nunique())
print("Negative Weekly_Sales rows:", (train["Weekly_Sales"] < 0).sum())
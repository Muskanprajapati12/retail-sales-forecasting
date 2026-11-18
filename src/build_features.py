# src/build_features.py
# Builds lag, rolling and calendar features from the weekly store sales.

import pandas as pd

# Load the weekly store-level data created with SQL
df = pd.read_csv("data/processed/weekly_store_sales.csv", parse_dates=["Date"])

# IMPORTANT: sort by store and date, so lags are computed in the right time order
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

# Group by store, so one store's sales never leaks into another store's lags
by_store = df.groupby("Store")["Weekly_Sales"]

# ---------- Lag features (past sales only) ----------
df["lag_1"] = by_store.shift(1)     # sales 1 week ago
df["lag_2"] = by_store.shift(2)     # sales 2 weeks ago
df["lag_4"] = by_store.shift(4)     # sales 4 weeks ago
df["lag_52"] = by_store.shift(52)   # sales in the same week last year

# ---------- Rolling averages (shift(1) first, to avoid leakage) ----------
df["rolling_mean_4"] = by_store.transform(lambda s: s.shift(1).rolling(4).mean())
df["rolling_mean_12"] = by_store.transform(lambda s: s.shift(1).rolling(12).mean())

# ---------- Calendar features (known in advance) ----------
df["month"] = df["Date"].dt.month
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

# ---------- Store features (fixed, known in advance) ----------
df["type_code"] = df["Type"].map({"A": 0, "B": 1, "C": 2})

# Drop columns that are NOT known at forecast time (leakage risk)
df = df.drop(columns=["Temperature", "Fuel_Price", "CPI", "Unemployment", "Type"])

# The first 52 weeks of every store have no lag_52, so drop rows with missing values
df = df.dropna().reset_index(drop=True)

# Save the final feature table
df.to_csv("data/processed/features_ready.csv", index=False)

# Sanity checks
print("Shape:", df.shape)
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
print("Weeks per store:", df.groupby("Store").size().iloc[0])
print("Missing values:", df.isnull().sum().sum())
print(df.columns.tolist())
print(df[df["Store"] == 1].head(3).T)
# src/check_store_14.py
# Compares Store 14 with the other stores to understand why its error is large.

import pandas as pd

df = pd.read_csv("data/processed/features_ready.csv", parse_dates=["Date"])

summary = df.groupby("Store").agg(
    avg_sales=("Weekly_Sales", "mean"),
    std_sales=("Weekly_Sales", "std"),
    size=("Size", "first"),
)
summary["variation_%"] = (summary["std_sales"] / summary["avg_sales"] * 100).round(1)

print("Store 14:")
print(summary.loc[14].round(1))
print("\nAverage of all stores:")
print(summary.mean().round(1))
print("\nRank of Store 14 by average sales (1 = biggest):",
      int(summary["avg_sales"].rank(ascending=False).loc[14]))
print("Rank of Store 14 by variation (1 = most unstable):",
      int(summary["variation_%"].rank(ascending=False).loc[14]))
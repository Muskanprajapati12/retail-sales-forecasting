# src/split_data.py
# Time-based train-test split (NO random shuffle) with leakage checks.

import pandas as pd

df = pd.read_csv("data/processed/features_ready.csv", parse_dates=["Date"])

# Number of most recent weeks kept for testing
TEST_WEEKS = 18

# Find the cutoff date from the sorted list of unique dates
all_dates = sorted(df["Date"].unique())
test_start = all_dates[-TEST_WEEKS]

# Everything before the cutoff is train, everything from the cutoff onwards is test
train = df[df["Date"] < test_start].copy()
test = df[df["Date"] >= test_start].copy()

# ---------- Leakage checks ----------
# 1) All train dates must be strictly earlier than all test dates
assert train["Date"].max() < test["Date"].min(), "Leakage: train and test dates overlap!"

# 2) Every store must appear in both train and test
assert set(train["Store"]) == set(test["Store"]), "Some store is missing in train or test!"

# 3) The target column must not be used as a feature
print("Target column is Weekly_Sales, features are everything except Store, Date and Weekly_Sales")

print("Train:", train.shape, "|", train["Date"].min().date(), "to", train["Date"].max().date())
print("Test :", test.shape, "|", test["Date"].min().date(), "to", test["Date"].max().date())
print("Train weeks:", train["Date"].nunique(), "| Test weeks:", test["Date"].nunique())
print("All leakage checks passed")

# Save both parts
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
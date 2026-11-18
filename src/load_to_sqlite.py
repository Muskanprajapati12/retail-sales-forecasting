# src/load_to_sqlite.py
# Loads the three raw CSV files into a SQLite database.

import sqlite3
import pandas as pd

DB_PATH = "data/processed/retail.db"

# Read the raw CSV files
train = pd.read_csv("data/raw/train.csv")
features = pd.read_csv("data/raw/features.csv")
stores = pd.read_csv("data/raw/stores.csv")

# Convert True/False to 1/0 so SQL can do MAX() and SUM() on it easily
train["IsHoliday"] = train["IsHoliday"].astype(int)
features["IsHoliday"] = features["IsHoliday"].astype(int)

# Connect to the database (the file is created if it does not exist)
con = sqlite3.connect(DB_PATH)

# Write each DataFrame as a table (replace it if it already exists)
train.to_sql("train", con, if_exists="replace", index=False)
features.to_sql("features", con, if_exists="replace", index=False)
stores.to_sql("stores", con, if_exists="replace", index=False)

# Check: count rows in every table using SQL
for table in ["train", "features", "stores"]:
    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"Table '{table}' loaded with {count} rows")

con.close()
print("Database saved at:", DB_PATH)
# src/weekly_sales_sql.py
# Uses SQL (JOIN + GROUP BY) to build weekly sales per store.

import sqlite3
import pandas as pd

DB_PATH = "data/processed/retail.db"

# SQL query: department-level rows -> one row per store per week
QUERY = """
SELECT
    t.Store                     AS Store,
    t.Date                      AS Date,
    SUM(t.Weekly_Sales)         AS Weekly_Sales,
    MAX(t.IsHoliday)            AS IsHoliday,
    s.Type                      AS Type,
    s.Size                      AS Size,
    f.Temperature               AS Temperature,
    f.Fuel_Price                AS Fuel_Price,
    f.CPI                       AS CPI,
    f.Unemployment              AS Unemployment
FROM train AS t
JOIN stores AS s
    ON t.Store = s.Store
JOIN features AS f
    ON t.Store = f.Store AND t.Date = f.Date
GROUP BY
    t.Store, t.Date, s.Type, s.Size,
    f.Temperature, f.Fuel_Price, f.CPI, f.Unemployment
ORDER BY
    t.Store, t.Date;
"""

con = sqlite3.connect(DB_PATH)

# Run the query and get the result as a DataFrame
weekly = pd.read_sql_query(QUERY, con)

# Save the result as a new table and also as a CSV
weekly.to_sql("weekly_store_sales", con, if_exists="replace", index=False)
weekly.to_csv("data/processed/weekly_store_sales.csv", index=False)
con.close()

# Sanity checks
print("Shape:", weekly.shape)
print("Stores:", weekly["Store"].nunique())
print("Weeks per store (min, max):",
      weekly.groupby("Store").size().min(),
      weekly.groupby("Store").size().max())
print("Missing values:")
print(weekly.isnull().sum())
print(weekly.head())
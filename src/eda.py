# src/eda.py
# Creates 3 EDA charts (trend, seasonality, holiday effect) and saves them in images/.

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts as files, do not open windows
import matplotlib.pyplot as plt

# Load the weekly store-level data created with SQL in Step 3
df = pd.read_csv("data/processed/weekly_store_sales.csv", parse_dates=["Date"])

# ---------- Chart 1: Trend ----------
# Total sales of all stores per week, in millions
total = df.groupby("Date")["Weekly_Sales"].sum() / 1e6

plt.figure(figsize=(12, 4))
plt.plot(total.index, total.values, label="Weekly total sales", alpha=0.6)
plt.plot(total.index, total.rolling(13).mean(), color="red", linewidth=2,
         label="13-week rolling average")
plt.title("Total Weekly Sales (all 45 stores)")
plt.ylabel("Sales (millions)")
plt.legend()
plt.savefig("images/eda_trend.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Chart 2: Seasonality ----------
df["Month"] = df["Date"].dt.month
monthly = df.groupby("Month")["Weekly_Sales"].mean() / 1e6

plt.figure(figsize=(8, 4))
plt.bar(monthly.index, monthly.values, color="steelblue")
plt.title("Average Weekly Sales per Store, by Month")
plt.xlabel("Month")
plt.ylabel("Sales (millions)")
plt.savefig("images/eda_seasonality.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Chart 3: Holiday effect ----------
holiday_avg = df.groupby("IsHoliday")["Weekly_Sales"].mean()
diff = (holiday_avg[1] / holiday_avg[0] - 1) * 100

plt.figure(figsize=(5, 4))
plt.bar(["Normal week", "Holiday week"], holiday_avg.values, color=["gray", "red"])
plt.title("Average Weekly Sales per Store")
plt.ylabel("Sales")
plt.savefig("images/eda_holiday.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Text summary ----------
print("Average sales, normal week :", round(holiday_avg[0]))
print("Average sales, holiday week:", round(holiday_avg[1]))
print(f"Holiday weeks are {diff:.1f}% different from normal weeks")
print("Best month on average:", monthly.idxmax())
print("Charts saved in the images folder")
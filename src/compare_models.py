# src/compare_models.py
# Builds the final comparison table and charts for the README.

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts as files, do not open windows
import matplotlib.pyplot as plt

# ---------- 1) Results table ----------
results = pd.read_csv("data/processed/results.csv")

# Improvement over the "Naive (last week)" baseline
base = results[results["Model"] == "Naive (last week)"].iloc[0]
results["MAE_improvement_%"] = ((1 - results["MAE"] / base["MAE"]) * 100).round(1)
results["RMSE_improvement_%"] = ((1 - results["RMSE"] / base["RMSE"]) * 100).round(1)
print(results.to_string(index=False))

# Save a markdown table that can be pasted directly into the README
lines = ["| Model | MAE | RMSE | MAE improvement vs naive | RMSE improvement vs naive |",
         "|---|---|---|---|---|"]
for _, r in results.iterrows():
    lines.append(f"| {r['Model']} | {r['MAE']:,} | {r['RMSE']:,} | "
                 f"{r['MAE_improvement_%']}% | {r['RMSE_improvement_%']}% |")
with open("data/processed/results_table.md", "w") as f:
    f.write("\n".join(lines))

# ---------- 2) Bar chart: MAE and RMSE for every model ----------
ax = results.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", figsize=(9, 4),
                                                      color=["steelblue", "darkorange"])
ax.set_title("Model Comparison on Test Set (lower is better)")
ax.set_ylabel("Error (dollars)")
plt.xticks(rotation=20, ha="right")
plt.savefig("images/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 3) Actual vs predicted ----------
test = pd.read_csv("data/processed/test.csv", parse_dates=["Date"])
rf = pd.read_csv("data/processed/pred_random_forest.csv", parse_dates=["Date"])
xgb = pd.read_csv("data/processed/pred_xgboost.csv", parse_dates=["Date"])

df = test[["Store", "Date", "Weekly_Sales", "lag_1"]].rename(columns={"lag_1": "pred_naive"})
df = df.merge(rf[["Store", "Date", "pred_random_forest"]], on=["Store", "Date"])
df = df.merge(xgb[["Store", "Date", "pred_xgboost"]], on=["Store", "Date"])


def plot_actual_vs_pred(data, title, path):
    plt.figure(figsize=(11, 4))
    plt.plot(data["Date"], data["Weekly_Sales"] / 1e6, label="Actual", color="black", linewidth=2)
    plt.plot(data["Date"], data["pred_naive"] / 1e6, label="Naive", linestyle=":", color="gray")
    plt.plot(data["Date"], data["pred_random_forest"] / 1e6, label="Random Forest", color="steelblue")
    plt.plot(data["Date"], data["pred_xgboost"] / 1e6, label="XGBoost", color="darkorange")
    plt.title(title)
    plt.ylabel("Sales (millions)")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# Chart A: total of all 45 stores per week
total = df.groupby("Date")[["Weekly_Sales", "pred_naive", "pred_random_forest", "pred_xgboost"]].sum().reset_index()
plot_actual_vs_pred(total, "Actual vs Predicted: Total of All Stores (Test Period)",
                    "images/forecast_total.png")

# Chart B: one single store
store_id = 1
one_store = df[df["Store"] == store_id]
plot_actual_vs_pred(one_store, f"Actual vs Predicted: Store {store_id} (Test Period)",
                    "images/forecast_store_1.png")

# ---------- 4) Which stores are hardest to forecast? ----------
df["xgb_abs_error"] = (df["Weekly_Sales"] - df["pred_xgboost"]).abs()
store_error = df.groupby("Store")["xgb_abs_error"].mean().sort_values(ascending=False)
print("\nTop 5 hardest stores for XGBoost (MAE):")
print(store_error.head(5).round(0))
print("\nCharts and table saved")
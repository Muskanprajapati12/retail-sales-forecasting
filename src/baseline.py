# src/baseline.py
# Naive baselines: no training, just copy a past value as the forecast.

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the time-based test set created in Step 6
test = pd.read_csv("data/processed/test.csv", parse_dates=["Date"])

y_true = test["Weekly_Sales"]


def evaluate(name, y_true, y_pred):
    """Return MAE and RMSE for one model as a dictionary."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:<28} MAE: {mae:>10,.0f}   RMSE: {rmse:>10,.0f}")
    return {"Model": name, "MAE": round(mae), "RMSE": round(rmse)}


# Baseline 1: forecast = sales of the previous week
naive_pred = test["lag_1"]

# Baseline 2: forecast = sales of the same week last year
seasonal_pred = test["lag_52"]

results = [
    evaluate("Naive (last week)", y_true, naive_pred),
    evaluate("Seasonal naive (last year)", y_true, seasonal_pred),
]

print("Average weekly sales in test set:", round(y_true.mean()))

# Save results so we can compare all models together later
pd.DataFrame(results).to_csv("data/processed/results.csv", index=False)
print("Results saved to data/processed/results.csv")
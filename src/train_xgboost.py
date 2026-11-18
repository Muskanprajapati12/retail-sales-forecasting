# src/train_xgboost.py
# Trains an XGBoost model on the train set and evaluates it on the time-based test set.

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts as files, do not open windows
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from config import FEATURES, TARGET

# Load the train and test sets created in Step 6
train = pd.read_csv("data/processed/train.csv", parse_dates=["Date"])
test = pd.read_csv("data/processed/test.csv", parse_dates=["Date"])

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# Fixed, sensible settings (NOT tuned on the test set)
model = XGBRegressor(
    n_estimators=300,       # number of boosting rounds (trees)
    learning_rate=0.05,     # small steps = more stable learning
    max_depth=4,            # shallow trees reduce overfitting
    subsample=0.8,          # each tree sees 80% of the rows
    colsample_bytree=0.8,   # each tree sees 80% of the features
    random_state=42,        # same result every run
    n_jobs=-1,              # use all CPU cores
)

# Learn from the past only
model.fit(X_train, y_train)

# Forecast the future (test period)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"XGBoost   MAE: {mae:,.0f}   RMSE: {rmse:,.0f}")

# ---------- Feature importance ----------
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
print("\nFeature importance (higher = more useful):")
print(importance.sort_values(ascending=False).round(3))

plt.figure(figsize=(7, 4))
plt.barh(importance.index, importance.values, color="darkorange")
plt.title("XGBoost Feature Importance")
plt.savefig("images/xgb_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Save model, predictions and score ----------
joblib.dump(model, "models/xgboost.pkl")

out = test[["Store", "Date", "Weekly_Sales"]].copy()
out["pred_xgboost"] = pred
out.to_csv("data/processed/pred_xgboost.csv", index=False)

results = pd.read_csv("data/processed/results.csv")
results = results[results["Model"] != "XGBoost"]  # avoid duplicate rows on re-run
new_row = pd.DataFrame([{"Model": "XGBoost", "MAE": round(mae), "RMSE": round(rmse)}])
pd.concat([results, new_row], ignore_index=True).to_csv("data/processed/results.csv", index=False)
print("\nSaved: model, predictions, feature chart and results")
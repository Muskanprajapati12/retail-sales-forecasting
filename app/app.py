# app/app.py
# Streamlit app: pick a store and view its sales forecast.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Folder with the processed CSV files (works no matter where you run the app from)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")


@st.cache_data
def load_data():
    """Load history and model forecasts. Cached so the files are read only once."""
    history = pd.read_csv(DATA_DIR / "features_ready.csv", parse_dates=["Date"])
    history = history[["Store", "Date", "Weekly_Sales"]]

    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["Date"])
    test = test[["Store", "Date", "Weekly_Sales", "lag_1"]].rename(columns={"lag_1": "Naive"})

    rf = pd.read_csv(DATA_DIR / "pred_random_forest.csv", parse_dates=["Date"])
    rf = rf[["Store", "Date", "pred_random_forest"]].rename(
        columns={"pred_random_forest": "Random Forest"})

    xgb = pd.read_csv(DATA_DIR / "pred_xgboost.csv", parse_dates=["Date"])
    xgb = xgb[["Store", "Date", "pred_xgboost"]].rename(columns={"pred_xgboost": "XGBoost"})

    forecasts = test.merge(rf, on=["Store", "Date"]).merge(xgb, on=["Store", "Date"])
    return history, forecasts


history, forecasts = load_data()

# ---------- Title ----------
st.title("Retail Sales Forecasting")
st.write("Weekly sales forecast for Walmart stores. Models are trained on past weeks only "
         "and tested on the most recent 18 weeks (time-based split).")

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
store = st.sidebar.selectbox("Select a store", sorted(forecasts["Store"].unique()))
model_options = ["Naive", "Random Forest", "XGBoost"]
models = st.sidebar.multiselect("Models to show", model_options, default=model_options)

# ---------- Data for the selected store ----------
cutoff = forecasts["Date"].min()  # first date of the test period
hist_store = history[(history["Store"] == store) & (history["Date"] < cutoff)]
fc_store = forecasts[forecasts["Store"] == store].sort_values("Date")

# ---------- Chart ----------
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(hist_store["Date"], hist_store["Weekly_Sales"] / 1e6,
        color="gray", label="History (training period)")
ax.plot(fc_store["Date"], fc_store["Weekly_Sales"] / 1e6,
        color="black", linewidth=2, label="Actual (test period)")

colors = {"Naive": "gray", "Random Forest": "steelblue", "XGBoost": "darkorange"}
for name in models:
    style = ":" if name == "Naive" else "-"
    ax.plot(fc_store["Date"], fc_store[name] / 1e6,
            color=colors[name], linestyle=style, label=name)

ax.axvline(cutoff, color="red", linestyle="--", alpha=0.5)
ax.set_title(f"Store {store}: Weekly Sales and Forecast")
ax.set_ylabel("Sales (millions)")
ax.legend()
st.pyplot(fig)

# ---------- Error metrics for this store ----------
st.subheader(f"Forecast error for Store {store} (MAE, lower is better)")
columns = st.columns(len(model_options))
for col, name in zip(columns, model_options):
    mae = (fc_store["Weekly_Sales"] - fc_store[name]).abs().mean()
    col.metric(name, f"${mae:,.0f}")

st.caption(f"Average weekly sales of Store {store} in the test period: "
           f"${fc_store['Weekly_Sales'].mean():,.0f}")

# ---------- Data table ----------
with st.expander("Show forecast table"):
    table = fc_store.copy()
    number_columns = ["Weekly_Sales", "Naive", "Random Forest", "XGBoost"]
    table[number_columns] = table[number_columns].round(0)  # round only the numeric columns
    table["Date"] = table["Date"].dt.date                   # show date without time
    st.dataframe(table, width="stretch")
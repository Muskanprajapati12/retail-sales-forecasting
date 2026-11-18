# src/config.py
# Shared settings used by all training scripts and the Streamlit app.

TARGET = "Weekly_Sales"

# Features known at forecast time (no future information, no store ID)
FEATURES = [
    "IsHoliday",
    "Size",
    "type_code",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_12",
    "month",
    "week_of_year",
]
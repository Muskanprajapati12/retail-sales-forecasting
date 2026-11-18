# src/check_setup.py
# This file only checks that all libraries are installed correctly.

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import sklearn
import xgboost
import streamlit

print("Setup check successful!")
print("pandas       :", pd.__version__)
print("numpy        :", np.__version__)
print("matplotlib   :", matplotlib.__version__)
print("seaborn      :", sns.__version__)
print("scikit-learn :", sklearn.__version__)
print("xgboost      :", xgboost.__version__)
print("streamlit    :", streamlit.__version__)
print("sqlite3      :", sqlite3.sqlite_version)
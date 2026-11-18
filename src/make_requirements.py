# src/make_requirements.py
# Writes a clean requirements.txt with only the libraries this project uses.

from importlib.metadata import version

PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "xgboost",
    "streamlit",
    "joblib",
]

with open("requirements.txt", "w", encoding="utf-8") as f:
    for name in PACKAGES:
        f.write(f"{name}=={version(name)}\n")

print(open("requirements.txt", encoding="utf-8").read())
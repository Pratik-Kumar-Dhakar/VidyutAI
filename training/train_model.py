
# train_model.py
# Usage: python train_model.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# 1. Paths
DATA_PATH = Path("./ev_battery_charging_data.csv")  # change path if needed
OUT_DIR = Path("./model")
OUT_DIR.mkdir(exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

# 3. Columns for modeling
num_cols = [
    "SOC (%)", "Voltage (V)", "Current (A)", "Battery Temp (°C)",
    "Ambient Temp (°C)", "Charging Duration (min)",
    "Degradation Rate (%)", "Efficiency (%)", "Charging Cycles"
]
cat_cols = ["Charging Mode", "Battery Type", "EV Model"]

# 4. Clean dataset
df = df[num_cols + cat_cols].dropna().reset_index(drop=True)

# 5. Encode categorical and scale numeric
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_arr = ohe.fit_transform(df[cat_cols])

scaler = StandardScaler()
num_arr = scaler.fit_transform(df[num_cols])

X = np.hstack([num_arr, cat_arr])

# 6. Train IsolationForest (unsupervised anomaly detection)
iso = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=0.1,
    random_state=42
)
iso.fit(X)

# 7. Compute data-driven thresholds using percentiles
thresholds = {}
for c in num_cols:
    series = df[c]
    thresholds[c] = {
        "p1": float(np.percentile(series, 1)),
        "p5": float(np.percentile(series, 5)),
        "p25": float(np.percentile(series, 25)),
        "p50": float(np.percentile(series, 50)),
        "p75": float(np.percentile(series, 75)),
        "p95": float(np.percentile(series, 95)),
        "p99": float(np.percentile(series, 99))
    }

# 8. Recommended actions for alerts (for backend enrichment)
actions = {
    "Battery Temp (°C)": {
        "high_action": "Reduce charging current / enable cooling",
        "explain": "High battery temperature accelerates degradation and can be unsafe."
    },
    "SOC (%)": {
        "low_action": "Stop discharge; schedule charging",
        "high_action": "Avoid pushing to 100% frequently to extend battery life"
    },
    "Voltage (V)": {
        "high_action": "Check for over-voltage condition; reduce charger voltage",
        "low_action": "Check connections; possible under-voltage"
    },
    "Degradation Rate (%)": {
        "high_action": "Schedule BMS diagnostics; plan battery replacement",
        "explain": "High degradation indicates aging cells / misuse."
    },
    "Efficiency (%)": {
        "low_action": "Investigate charging system / power electronics"
    },
    "Current (A)": {
        "high_action": "Reduce charging rate; enable cooling"
    }
}

# 9. Save trained artifacts
joblib.dump(iso, OUT_DIR / "isolation_forest.joblib")
joblib.dump(ohe, OUT_DIR / "ohe.joblib")
joblib.dump(scaler, OUT_DIR / "scaler.joblib")
with open(OUT_DIR / "thresholds.json", "w") as f:
    json.dump({"num_cols": num_cols, "thresholds": thresholds, "actions": actions}, f, indent=2)

print("✅ Model and thresholds saved to ./model")

#!/Users/advait/Code/hackathons/vidyut_ai_2025/backend/a/bin/python3
# inference.py
import sys, json, numpy as np, pandas as pd
from pathlib import Path
import joblib

MODEL_DIR = Path("./model")
iso = joblib.load(MODEL_DIR / "isolation_forest.joblib")
ohe = joblib.load(MODEL_DIR / "ohe.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
with open(MODEL_DIR / "thresholds.json", "r") as f:
    meta = json.load(f)

num_cols = meta["num_cols"]
thresholds = meta["thresholds"]
actions_map = meta.get("actions", {})

def feature_threshold_checks(feat):
    alerts = []
    for k, v in feat.items():
        if k not in thresholds or v is None: continue
        t = thresholds[k]
        if v > t["p99"]:
            alerts.append({"feature": k, "level": "critical_high", "value": v})
        elif v > t["p95"]:
            alerts.append({"feature": k, "level": "high", "value": v})
        if v < t["p1"]:
            alerts.append({"feature": k, "level": "critical_low", "value": v})
        elif v < t["p5"]:
            alerts.append({"feature": k, "level": "low", "value": v})
    return alerts

def encode_and_predict(df_row):
    num_df = df_row[num_cols].astype(float)

    # Get the categorical columns in the same order used during training
    cat_cols = list(ohe.feature_names_in_) if hasattr(ohe, "feature_names_in_") else []
    
    # Ensure all expected categorical columns are present
    for c in cat_cols:
        if c not in df_row.columns:
            df_row[c] = None

    cat_input = df_row[cat_cols]  # keep exact order

    # Transform numeric + categorical features
    X_num = scaler.transform(num_df)
    X_cat = ohe.transform(cat_input) if cat_cols else np.empty((len(df_row), 0))
    X = np.hstack([X_num, X_cat])

    score = iso.decision_function(X)
    pred = iso.predict(X)
    return pred, score
# Read JSON from stdin (from Node)
raw = sys.stdin.read()
try:
    payload = json.loads(raw)
    d = payload.get("data", {})
    df_row = pd.DataFrame([d], columns=list(list(d.keys()) + num_cols))
    for c in num_cols:
        if c not in df_row.columns:
            df_row[c] = float("nan")

    try:
        pred, score = encode_and_predict(df_row)
    except Exception as e:
        feat = {c: float(d.get(c)) if d.get(c) is not None else None for c in num_cols}
        alerts = feature_threshold_checks(feat)
        print(json.dumps({"ok": False, "error": str(e), "alerts": alerts}))
        sys.exit(0)

    feat = {c: float(d.get(c)) if d.get(c) is not None else None for c in num_cols}
    alerts = feature_threshold_checks(feat)

    is_anomaly = int(pred[0] == -1)
    if is_anomaly:
        alerts.append({
            "feature": "multivariate",
            "level": "anomaly",
            "value": float(score[0]),
            "desc": "Multivariate anomaly detected by IsolationForest"
        })

    enriched_alerts = []
    for a in alerts:
        feature = a.get("feature")
        rec = {}
        if feature in actions_map:
            rec = actions_map[feature]
        elif feature == "multivariate":
            rec = {"recommendation": "Investigate system; run full BMS and power electronics diagnostics."}
        else:
            rec = {"recommendation": "Review telemetry and escalate to maintenance."}

        if a.get("level", "").endswith("high") and "high_action" in rec:
            a["recommended_action"] = rec["high_action"]
        elif a.get("level", "").endswith("low") and "low_action" in rec:
            a["recommended_action"] = rec["low_action"]
        else:
            a["recommended_action"] = rec.get("recommendation", rec.get("explain", ""))

        enriched_alerts.append(a)

    print(json.dumps({
        "ok": True,
        "anomaly_score": float(score[0]),
        "is_anomaly": bool(is_anomaly),
        "alerts": enriched_alerts
    }))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))

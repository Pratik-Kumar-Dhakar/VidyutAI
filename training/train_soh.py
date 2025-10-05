
# battery_soh_gpr.py
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 1. LOAD DATA
data_file = "ev_battery_charging_data.csv"
df = pd.read_csv(data_file)

# 2. ENGINEER TARGET (SoH, if not already present)
if 'Degradation Rate (%)' in df.columns:
    df['SoH'] = (100 - df['Degradation Rate (%)']) / 100
# If 'SoH' column already present, skip above.

# 3. FEATURE ENGINEERING: update columns as per your CSV
features = [
    'Voltage', 'Current', 'Battery Temperature',
    'Ambient Temperature', 'SOC (%)',
    'Charging Duration', 'Efficiency (%)',
    # Add categorical feature encoding if needed:
    # 'Battery Type', 'Charging Mode', 'EV Model'
    # If these are strings, use pd.get_dummies(df[...])
]

# Encode categoricals if present
categoricals = []
for col in ['Battery Type', 'Charging Mode', 'EV Model']:
    if col in df.columns:
        categoricals.append(col)
if categoricals:
    df = pd.get_dummies(df, columns=categoricals)

selected_features = features + [
    c for c in df.columns if c.startswith(tuple(categoricals))
]

# 4. SPLIT DATA
print(selected_features)
print(df.columns)
X = df[selected_features].copy()
y = df['SoH']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# 5. SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. MODEL TRAINING
kernel = 1.0 * RBF(length_scale=1.0) + 1.0 * Matern(length_scale=1.0, nu=2.5) + 1.0 * RationalQuadratic()
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=5, normalize_y=True, random_state=42)

gpr.fit(X_train_scaled, y_train)

# 7. MODEL EVALUATION
y_pred, y_std = gpr.predict(X_test_scaled, return_std=True)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("Evaluation Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# 8. SAVE MODEL + SCALER
model_data = {
    "model": gpr,
    "scaler": scaler,
    "features": selected_features
}
joblib.dump(model_data, "ev_battery_soh_gpr_model.pkl")

print("\nModel trained and saved as 'ev_battery_soh_gpr_model.pkl'.")

# 9. PREDICT NEW DATA EXAMPLE
def predict_soh(new_data_csv):
    input_df = pd.read_csv(new_data_csv)
    # Handle categoricals and missing columns as above
    for col in categoricals:
        if col in input_df.columns:
            input_df = pd.get_dummies(input_df, columns=[col])
    # Filter and align columns
    input_X = input_df[model_data["features"]]
    input_X_scaled = model_data["scaler"].transform(input_X)
    soh_pred, soh_std = model_data["model"].predict(input_X_scaled, return_std=True)
    return soh_pred, soh_std

# Example usage:
# soh_pred, soh_std = predict_soh("new_charging_data.csv")
# print(soh_pred, soh_std)

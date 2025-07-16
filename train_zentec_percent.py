# train_zentec_percent.py

import pandas as pd
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load and clean data
data = pd.read_csv("zentec_stock_data.csv", header=2, index_col="Date", parse_dates=True)
data.columns = ["Close", "High", "Low", "Open", "Volume"]

# Add technical indicators
data["ma5"] = ta.trend.sma_indicator(data["Close"], window=5)
data["ema10"] = ta.trend.ema_indicator(data["Close"], window=10)
data["rsi14"] = ta.momentum.rsi(data["Close"], window=14)

data.dropna(inplace=True)

# Features
X = data[["ma5", "ema10", "rsi14"]]

# Target: Predict % change
y = (data["Close"].shift(-1) - data["Close"]) / data["Close"]

# Drop last row where y is NaN
X = X[:-1]
y = y[:-1]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"📉 MSE on % return prediction: {mse:.6f}")

# Save model
joblib.dump(model, "anscom_zentec_model_percent.pkl")
print("✅ Model saved as 'anscom_zentec_model_percent.pkl'")

# predict_zentec_percent.py

import pandas as pd
import yfinance as yf
import joblib
import ta

# Step 1: Here i'll download 60 days of live data
ticker = "ZENTEC.NS"
data = yf.download(ticker, period="60d", interval="1d")

# Step 2: i would like to Keep required columns
data = data[["Close", "High", "Low", "Open", "Volume"]]
data.dropna(inplace=True)

# Step 3: now let's Add indicators
close = data["Close"].squeeze()  # Ensure 1D
data["ma5"] = ta.trend.sma_indicator(close, window=5)
data["ema10"] = ta.trend.ema_indicator(close, window=10)
data["rsi14"] = ta.momentum.rsi(close, window=14)
data.dropna(inplace=True)

# Step 4: Prepare latest input features
latest_data = data[["ma5", "ema10", "rsi14"]].iloc[-1:]
current_price = float(data["Close"].iloc[-1])  # Make sure it's a number

# Step 5: now we could load trained model
model = joblib.load("anscom_zentec_model_percent.pkl")

# Step 6: finally the commands to Predict
predicted_pct_change = model.predict(latest_data)[0]
predicted_price = current_price * (1 + predicted_pct_change)

# Step 7: Output and some comments for our confirmation
print(f"📆 Date: {data.index[-1].date()}")
print(f"📉 Current Price: ₹{current_price:.2f}")
print(f"📈 Predicted % Change: {predicted_pct_change * 100:.2f}%")
print(f"🔮 Predicted Next Price: ₹{predicted_price:.2f}")

# Step 8: Signal
if predicted_pct_change > 0.01:
    print("🟢 Signal: BUY")
elif predicted_pct_change < -0.01:
    print("🔴 Signal: SELL")
else:
    print("🟡 Signal: HOLD")


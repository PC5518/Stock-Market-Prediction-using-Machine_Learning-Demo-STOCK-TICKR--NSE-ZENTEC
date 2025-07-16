# 📈 NSE: ZenTech Stock Prediction using Machine Learning
REGISTERED ON NATIONAL STOCK EXCHANGE OF INDIA 
A lightweight and practical machine learning project to predict **next-day percentage price changes** for Zen Technologies (ZENTEC.NS) stock using historical price trends and technical indicators.

## 🚀 Key Features

- 📊 **Data Sourcing**: Automatically fetches historical stock data using Yahoo Finance.
- 📉 **Feature Engineering**: Applies SMA, EMA, and RSI indicators via `ta` library.
- 🧠 **Model**: Trained with RandomForestRegressor on technical features.
- 🔮 **Live Prediction**: Uses latest 60 days of stock data to predict next-day percentage change and price.
- ✅ **Signal Output**: Generates BUY / SELL / HOLD signals based on prediction threshold.

---

## 📁 Files Explained

### 1. `zentec_stock_data.csv`
Full historical stock data for ZenTech downloaded from Yahoo Finance.

### 2. `train_zentec_percent.py`
Trains a `RandomForestRegressor` model to predict **percentage price change** from technical indicators. Saves model to `.pkl`.

### 3. `predict_zentec_percent.py`
Loads live 60-day stock data, applies same indicators, and predicts next-day percentage change and price.

### 4. `anscom_zentec_model_percent.pkl`
Final trained model. Accurate and production-ready. Earlier versions like `_rf` were experimental.

---

## 🧠 Example Output

📆 Date: 2025-07-15
📉 Current Price: ₹872.30
📈 Predicted % Change: 2.41%
🔮 Predicted Next Price: ₹893.30
🟢 Signal: BUY

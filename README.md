# 📈 NSE: ZenTech Stock Prediction using Machine Learning
REGISTERED ON NATIONAL STOCK EXCHANGE OF INDIA 
A lightweight and practical machine learning project to predict **next-day percentage price changes** for Zen Technologies (ZENTEC.NS) stock using historical price trends and technical indicators.
## Concept of this Project :
This project explores how machine learning can be used to understand and predict stock market trends by learning from historical data. I’ve implemented and trained machine learning models to analyze price patterns and generate future predictions — combining core concepts of data science, time-series analysis, and real-world financial insights. The only key impact is that the earnings of the stock affect the market. That cannot be predicted by this code yet. basically this project used stuff like linear algebra to handle data in matrix form, statistics and probability to find patterns and analyze trends in the stock prices, calculus mainly for optimization like gradient descent to train the model, regression analysis like linear regression to actually predict the price, and time series analysis because stock data changes with time, so it looks at how things evolve day by day .(And all these things are basically built or indirectly used Or may be directly used by the Python machine learning modules like scikit-learn, etc)
## 🚀 Key Features

- 📊 **Data Sourcing**: Automatically fetches historical stock data using Yahoo Finance.
- 📉 **Feature Engineering**: Applies SMA, EMA, and RSI indicators via `ta` library.
- 🧠 **Model**: Trained with RandomForestRegressor on technical features.
- 🔮 **Live Prediction**: Uses latest 60 days of stock data to predict next-day percentage change and price.
- ✅ **Signal Output**: Generates BUY / SELL / HOLD signals based on prediction threshold.
## Modules used by me :
pandas – For handling, cleaning, and manipulating the market dataset (e.g., CSV data, time-series).

numpy – For numerical operations like vectorized calculations, matrix operations.

matplotlib / seaborn – For plotting data trends, price curves, and prediction visualizations.

scikit-learn – Used for:

Train-test splitting (train_test_split)

Machine learning models like LinearRegression, RandomForestRegressor, etc.

Metrics like mean_squared_error or r2_score

statsmodels – (Optional) For advanced time series modeling or statistical testing (like trend significance).

xgboost – (If used) For high-performance gradient boosting-based regression.

yfinance / pandas_datareader – For fetching real-time or historical market data.

datetime – Handling time-based data, timestamps, and feature engineering.

joblib / pickle – Saving and loading trained models.

tkinter / streamlit – (If GUI/web) For building a frontend/dashboard for predictions.

argparse – (Optional CLI usage) For command-line interface to pass arguments like ticker name, model name, etc
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
& HERE'S THE OUTPUT AND PREDICTION RESULT USING MACHINE LEARNING
<img width="1919" height="1017" alt="MODEL OUTPUT AND ACCURACY RESULT" src="https://github.com/user-attachments/assets/42fff2b9-b62e-4520-a792-3513e2787941" />


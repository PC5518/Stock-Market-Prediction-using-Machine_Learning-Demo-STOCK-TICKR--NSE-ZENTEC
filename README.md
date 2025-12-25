# NSE Equity Volatility & Valuation Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)

## 🏗 System Abstract
This repository houses a **quantitative predictive modeling framework** designed to forecast short-term price volatility for NSE-listed equities (specifically *Zen Technologies*). 

Unlike traditional price-target models, this system minimizes variance by training on **relative percentage returns** rather than absolute price values. It leverages an **Ensemble Learning architecture (Random Forest Regressor)** to detect non-linear dependencies between momentum indicators (RSI, SMA, EMA) and future price action, outputting a probabilistic **BUY/SELL/HOLD** signal based on a ±1% volatility threshold.

## 🚀 Key Technical Features
*   **Stochastic Data Pipeline:** Automated ingestion of OHLCV market data via `yfinance` APIs, supporting both historical backtesting (2000–Present) and real-time inference.
*   **Vectorized Feature Engineering:** utilizes `pandas` and the `ta` library to synthesize technical vectors:
    *   **SMA (5-day):** Trend smoothing for short-term signal detection.
    *   **EMA (10-day):** Weighted moving average to prioritize recent price action.
    *   **RSI (14-day):** Momentum oscillator to identify overbought/oversold conditions.
*   **Ensemble Regression:** Implements `RandomForestRegressor` (100 estimators) to mitigate overfitting common in single decision trees when applied to noisy financial time-series data.
*   **Risk-Adjusted Decision Logic:** Generates signals only when predicted volatility exceeds a specific confidence threshold (>1%), filtering out market noise.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `zentec_stock_data.csv` | **Data Layer:** Raw OHLCV dataset fetched from NSE (2000–2025). |
| `train_zentec_percent.py` | **Training Pipeline:** preprocessing, feature extraction, train-test splitting (80/20), and model serialization. |
| `predict_zentec_percent.py` | **Inference Engine:** Fetches live trailing 60-day data, regenerates features, and computes next-day directional probability. |
| `anscom_zentec_model_percent.pkl` | **Serialized Model:** The optimized Random Forest model artifact. |

## 🛠 Usage & Installation

### 1. Prerequisites
Ensure the quantitative stack is installed:
```bash
pip install pandas yfinance ta scikit-learn joblib

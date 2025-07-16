import yfinance as yf
import pandas as pd

# Define the ticker for Zen Technologies (NSE)
ticker = "ZENTEC.NS"

# Download data
data = yf.download(ticker, start="2000-01-01", end="2025-07-15")

# Save the data to CSV file locally
data.to_csv("zentec_stock_data.csv")

print("✅ Data downloaded and saved as 'zentec_stock_data.csv'")

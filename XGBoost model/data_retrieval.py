import pandas as pd
import numpy as np
import requests
from datetime import datetime


# Function to fetch data from Binance API
def get_crypto_data(symbol, interval, start_str, end_str):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(pd.to_datetime(start_str).timestamp() * 1000),
        "endTime": int(pd.to_datetime(end_str).timestamp() * 1000)
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
                                     "Quote asset volume", "Number of trades", "Taker buy base asset volume",
                                     "Taker buy quote asset volume", "Ignore"])
    df["Close"] = df["Close"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')
    return df


# Function to calculate technical indicators and metrics
def calculate_metrics(data, variable1, variable2):
    # RSI Calculation
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # EMA and SMA Calculation
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # VWAP Calculation
    data['Cumulative_TPV'] = (data['Close'] * data['Volume']).cumsum()
    data['Cumulative_Volume'] = data['Volume'].cumsum()
    data['VWAP'] = data['Cumulative_TPV'] / data['Cumulative_Volume']

    # TWAP Calculation
    data['TWAP'] = data['Close'].expanding().mean()

    # Historical High and Low Price Calculation
    data[f'High_Last_{variable1}_Days'] = data['High'].rolling(window=variable1).max()
    data[f'Days_Since_High_Last_{variable1}_Days'] = data['High'].rolling(window=variable1).apply(
        lambda x: variable1 - x[::-1].idxmax() if not x.isna().all() else np.nan)
    data[f'%_Diff_From_High_Last_{variable1}_Days'] = (data['Close'] - data[f'High_Last_{variable1}_Days']) / data[
        f'High_Last_{variable1}_Days'] * 100

    data[f'Low_Last_{variable1}_Days'] = data['Low'].rolling(window=variable1).min()
    data[f'Days_Since_Low_Last_{variable1}_Days'] = data['Low'].rolling(window=variable1).apply(
        lambda x: variable1 - x[::-1].idxmin() if not x.isna().all() else np.nan)
    data[f'%_Diff_From_Low_Last_{variable1}_Days'] = (data['Close'] - data[f'Low_Last_{variable1}_Days']) / data[
        f'Low_Last_{variable1}_Days'] * 100

    # Future High and Low Price Calculation
    data[f'High_Next_{variable2}_Days'] = data['High'].shift(-variable2).rolling(window=variable2).max()
    data[f'%_Diff_From_High_Next_{variable2}_Days'] = (data[f'High_Next_{variable2}_Days'] - data['Close']) / data[
        'Close'] * 100

    data[f'Low_Next_{variable2}_Days'] = data['Low'].shift(-variable2).rolling(window=variable2).min()
    data[f'%_Diff_From_Low_Next_{variable2}_Days'] = (data[f'Low_Next_{variable2}_Days'] - data['Close']) / data[
        'Close'] * 100

    # Drop rows with NaN values to clean up the dataset
    data = data.dropna()

    # Save to Excel as per assessment requirements
    with pd.ExcelWriter('crypto_data_wth_metric.xlsx') as writer:
        data.to_excel(writer, sheet_name='Data with Metrics', index=False)

    return data
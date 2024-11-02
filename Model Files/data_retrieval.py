import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_crypto_data(crypto_pair, start_date, end_date):
    """
    Fetch historical cryptocurrency data from the Binance API in chunks to handle date range limits.

    Parameters:
    - crypto_pair: The trading pair, e.g., 'BTCUSDT'.
    - start_date: The start date in 'YYYY-MM-DD' format.
    - end_date: The end date in 'YYYY-MM-DD' format.

    Returns:
    - A DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []

    # Convert dates to timestamps
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Fetch data in chunks
    while current_date <= end_date_dt:
        # Calculate the next chunk end date (maximum range per request is 1000 days)
        next_date = current_date + timedelta(days=1000)
        if next_date > end_date_dt:
            next_date = end_date_dt

        # Convert to timestamps
        start_timestamp = int(current_date.timestamp()) * 1000
        end_timestamp = int(next_date.timestamp()) * 1000

        # Fetch the data from the API
        params = {
            'symbol': crypto_pair,
            'interval': '1d',
            'startTime': start_timestamp,
            'endTime': end_timestamp
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.text}")

        data = response.json()
        all_data.extend(data)

        # Move to the next chunk
        current_date = next_date + timedelta(days=1)

    # Convert data to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
    ])

    # Select only the required columns and format them
    df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], unit='ms').dt.date
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    return df

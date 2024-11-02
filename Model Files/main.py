import data_retrieval
import ml_model
import pandas as pd
import pandas_ta as ta


def main():
    # Fetch historical crypto data with both start and end dates
    crypto_pair = "BTCUSDT"
    start_date = "2018-01-01"
    end_date = "2024-10-31"
    data = data_retrieval.fetch_crypto_data(crypto_pair, start_date, end_date)

    # Convert the Date column to a datetime object
    data['Date'] = pd.to_datetime(data['Date'])

    # === Feature Engineering ===
    # Calculate Volatility
    data['Volatility'] = data['Close'].rolling(window=14).std()

    # MACD: Moving Average Convergence Divergence
    macd = ta.macd(data['Close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['MACD_Signal'] = macd['MACDs_12_26_9']

    # RSI: Relative Strength Index
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # Bollinger Bands
    bb = ta.bbands(data['Close'], length=20)
    data['BB_Middle'] = bb.iloc[:, 0]  # Middle band
    data['BB_Upper'] = bb.iloc[:, 1]  # Upper band
    data['BB_Lower'] = bb.iloc[:, 2]  # Lower band

    # VWAP: Volume Weighted Average Price
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

    # Define the number of days for look-back and look-forward calculations
    variable1 = 7  # Look-back period
    variable2 = 5  # Look-forward period

    # Calculate historical and future metrics
    data[f'High_Last_{variable1}_Days'] = data['High'].rolling(window=variable1).max()
    data[f'Days_Since_High_Last_{variable1}_Days'] = (data['Date'] - data['Date'].shift(variable1)).dt.days
    data[f'%_Diff_From_High_Last_{variable1}_Days'] = (
        (data['Close'] - data[f'High_Last_{variable1}_Days']) / data[f'High_Last_{variable1}_Days'] * 100
    )
    data[f'Low_Last_{variable1}_Days'] = data['Low'].rolling(window=variable1).min()
    data[f'Days_Since_Low_Last_{variable1}_Days'] = (data['Date'] - data['Date'].shift(variable1)).dt.days
    data[f'%_Diff_From_Low_Last_{variable1}_Days'] = (
        (data['Close'] - data[f'Low_Last_{variable1}_Days']) / data[f'Low_Last_{variable1}_Days'] * 100
    )
    data[f'High_Next_{variable2}_Days'] = data['High'].shift(-variable2).rolling(window=variable2).max()
    data[f'%_Diff_From_High_Next_{variable2}_Days'] = (
        (data[f'High_Next_{variable2}_Days'] - data['Close']) / data['Close'] * 100
    )
    data[f'Low_Next_{variable2}_Days'] = data['Low'].shift(-variable2).rolling(window=variable2).min()
    data[f'%_Diff_From_Low_Next_{variable2}_Days'] = (
        (data[f'Low_Next_{variable2}_Days'] - data['Close']) / data['Close'] * 100
    )

    # Drop rows with NaN values created by rolling calculations
    data = data.dropna()

    # Train the model
    models, rmses = ml_model.train_model(data)
    print(f"XGBoost Model RMSE: {rmses[0]}")
    print(f"CatBoost Model RMSE: {rmses[1]}")

    # Example prediction using the trained models
    input_data = {
        f'Days_Since_High_Last_{variable1}_Days': 5,
        f'%_Diff_From_High_Last_{variable1}_Days': -2.5,
        f'Days_Since_Low_Last_{variable1}_Days': 3,
        f'%_Diff_From_Low_Last_{variable1}_Days': 1.5,
        'Volatility': data['Volatility'].iloc[-1],
        'MACD': data['MACD'].iloc[-1],
        'MACD_Signal': data['MACD_Signal'].iloc[-1],
        'RSI': data['RSI'].iloc[-1],
        'BB_Middle': data['BB_Middle'].iloc[-1],
        'BB_Upper': data['BB_Upper'].iloc[-1],
        'BB_Lower': data['BB_Lower'].iloc[-1],
        'VWAP': data['VWAP'].iloc[-1]
    }
    predictions = ml_model.predict_outcomes(models, input_data)
    print("Predictions:", predictions)

    # Save the final DataFrame to an Excel file
    data.to_excel("model_predictions.xlsx", index=False)
    print("Data saved to model_predictions.xlsx")

if __name__ == "__main__":
    main()

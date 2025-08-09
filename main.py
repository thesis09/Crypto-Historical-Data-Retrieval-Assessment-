from data_retrieval import get_crypto_data, calculate_metrics
from ml_model import preprocess_data, train_xgboost
import pandas as pd
import joblib

# Parameters
symbol = "BTCUSDT"
interval = "15m"  # 15 minutes interval
start_date = "2020-01-01"  # Extended start date for larger dataset
end_date = "2025-06-30"
variable1 = 30  # Look-back period for historical metrics
variable2 = 5  # Look-forward period for future metrics

# Step 1: Get Data
data = get_crypto_data(symbol, interval, start_date, end_date)

# Step 2: Calculate Metrics
data_with_metrics = calculate_metrics(data, variable1, variable2)

# Step 3: Define Features and Targets
features = [
    f'Days_Since_High_Last_{variable1}_Days',
    f'%_Diff_From_High_Last_{variable1}_Days',
    f'Days_Since_Low_Last_{variable1}_Days',
    f'%_Diff_From_Low_Last_{variable1}_Days',
    'RSI', 'EMA_20', 'SMA_50', 'VWAP', 'TWAP'
]
target_high = f'%_Diff_From_High_Next_{variable2}_Days'
target_low = f'%_Diff_From_Low_Next_{variable2}_Days'

# Step 4: Preprocess Data for High Price Prediction
X_train_high, X_test_high, y_train_high, y_test_high = preprocess_data(data_with_metrics, features, target_high)

# Step 5: Train XGBoost Model for High Price Prediction
xgb_model_high, xgb_metrics_high = train_xgboost(X_train_high, y_train_high, X_test_high, y_test_high)

# Step 6: Preprocess Data for Low Price Prediction
X_train_low, X_test_low, y_train_low, y_test_low = preprocess_data(data_with_metrics, features, target_low)

# Step 7: Train XGBoost Model for Low Price Prediction
xgb_model_low, xgb_metrics_low = train_xgboost(X_train_low, y_train_low, X_test_low, y_test_low)

# Step 8: Save Models and Metrics
joblib.dump(xgb_model_high, 'xgb_model_high.pkl')
joblib.dump(xgb_model_low, 'xgb_model_low.pkl')

with pd.ExcelWriter('model_etrics.xlsx') as writer:
    pd.DataFrame([xgb_metrics_high]).to_excel(writer, sheet_name='High Price Metrics', index=False)
    pd.DataFrame([xgb_metrics_low]).to_excel(writer, sheet_name='Low Price Metrics', index=False)

# Print Metrics
print("XGBoost High Price Model Metrics:", xgb_metrics_high)
print("XGBoost Low Price Model Metrics:", xgb_metrics_low)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

def train_model(data):
    # Extract features and target variables
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'MACD', 'MACD_Signal', 'RSI',
                'BB_Middle', 'BB_Upper', 'BB_Lower', 'VWAP', 'High_Last_7_Days', 'Days_Since_High_Last_7_Days',
                '%_Diff_From_High_Last_7_Days', 'Low_Last_7_Days', 'Days_Since_Low_Last_7_Days',
                '%_Diff_From_Low_Last_7_Days']
    targets = ['%_Diff_From_High_Next_5_Days', '%_Diff_From_Low_Next_5_Days']

    X = data[features]
    y = data[targets]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBoost Regressor (using CPU)
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='reg:squarederror',
        tree_method='auto',
        n_jobs=-1

    )

    # Train XGBoost using MultiOutputRegressor
    xgb_multi_model = MultiOutputRegressor(xgb_model)
    xgb_multi_model.fit(X_train, y_train)

    # Tune and train CatBoost for each target separately
    tuned_cat_models = []
    cat_rmse_values = []

    # Define a reduced hyperparameter grid for faster tuning
    param_grid = {
        'iterations': [500, 1000],
        'learning_rate': [0.01, 0.05],
        'depth': [4, 6],
        'l2_leaf_reg':[3, 5]
    }

    for i, target in enumerate(targets):
        # Initialize CatBoost Regressor (using GPU) with early stopping
        cat_model = CatBoostRegressor(
            task_type='GPU',
            devices='0',
            silent=True,
            early_stopping_rounds=50  # Early stopping for faster training

        )

        # Perform RandomizedSearchCV for hyperparameter tuning with fewer iterations
        random_search = RandomizedSearchCV(
            estimator=cat_model,
            param_distributions=param_grid,
            n_iter=5,  # Reduced number of parameter settings
            scoring='neg_mean_squared_error',
            cv=2,  # Simpler cross-validation for speed
            verbose=2,
            random_state=42
        )

        # Fit RandomizedSearchCV to the training data for the specific target
        random_search.fit(X_train, y_train.iloc[:, i])
        best_cat_model = random_search.best_estimator_

        # Append the tuned model and calculate RMSE
        tuned_cat_models.append(best_cat_model)
        cat_predictions = best_cat_model.predict(X_test)
        cat_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], cat_predictions))
        cat_rmse_values.append(cat_rmse)

    # Calculate RMSE for XGBoost
    xgb_predictions = xgb_multi_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions, multioutput='raw_values')).mean()

    print(f"XGBoost Model RMSE: {xgb_rmse}")
    print(f"Tuned CatBoost Model RMSEs: {cat_rmse_values}")
    print("Best CatBoost Models Tuned Separately")

    # Return both models and their RMSE values
    return (xgb_multi_model, tuned_cat_models), (xgb_rmse, cat_rmse_values)

def predict_outcomes(models, input_data):
    xgb_model, cat_models = models

    # Ensure the input data has the same features as the training data
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'MACD', 'MACD_Signal', 'RSI',
                     'BB_Middle', 'BB_Upper', 'BB_Lower', 'VWAP', 'High_Last_7_Days', 'Days_Since_High_Last_7_Days',
                     '%_Diff_From_High_Last_7_Days', 'Low_Last_7_Days', 'Days_Since_Low_Last_7_Days',
                     '%_Diff_From_Low_Last_7_Days']

    # Convert input data to a DataFrame with the correct feature names and order
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Make predictions
    xgb_predictions = xgb_model.predict(input_df)
    cat_predictions = [model.predict(input_df)[0] for model in cat_models]

    # Return predictions from both models
    return {
        'XGB_%_Diff_From_High_Next_5_Days': xgb_predictions[0][0],
        'XGB_%_Diff_From_Low_Next_5_Days': xgb_predictions[0][1],
        'Cat_%_Diff_From_High_Next_5_Days': cat_predictions[0],
        'Cat_%_Diff_From_Low_Next_5_Days': cat_predictions[1]
    }

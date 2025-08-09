### File: ml_model.py ###

import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor


# Function to preprocess data
def preprocess_data(data, features, target):
    X = data[features].dropna()
    y = data.loc[X.index, target]

    # Handle any remaining NaN values - replace them with mean (or you can use other methods)
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    return train_test_split(X, y, test_size=0.2, random_state=42)


# Function to train XGBoost model with cross-validation and hyperparameter tuning
def train_xgboost(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [500, 700, 900],  # Increase to better fit larger datasets
        'learning_rate': [0.01, 0.05, 0.1],  # Range to improve training dynamics
        'max_depth': [3,5, 7, 10],  # Explore deeper trees for high value prediction
        'min_child_weight': [3, 5],  # Adjust to improve model generalization
        'gamma': [0.1, 0.3],  # Moderate regularization to avoid overfitting
        'subsample': [0.8, 0.9],  # Control for overfitting with more variety
        'colsample_bytree': [0.8, 0.9],  # Feature subsampling
        'reg_alpha': [0.3, 0.5],  # Regularization strength to control model complexity
        'reg_lambda': [1, 2]  # Regularization penalty for model weights
    }

    # Configure model without specifying the deprecated predictor
    xgb_model = XGBRegressor(objective='reg:squarederror', tree_method='hist')

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=300,  # Moderate exploration without excessive runtime
        scoring='neg_mean_squared_error',
        cv=TimeSeriesSplit(n_splits=5),  # Time series split for temporal data
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Fit the model without early stopping (for CPU training compatibility)
    random_search.fit(X_train, y_train, verbose=True)

    # Predict using the best model
    y_pred = random_search.best_estimator_.predict(X_test)

    metrics = {
        #'rmse': root_mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred)
    }
    return random_search.best_estimator_, metrics

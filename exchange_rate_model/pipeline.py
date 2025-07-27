import pandas as pd
import numpy as np
from exchange_rate_model.data_loader import load_exchange_rate_from_db
from exchange_rate_model.feature_engineering import log_and_create_lags
from exchange_rate_model.forecast import forecast_next_week
from typing import Optional

def full_pipeline(database_path: str, table_name: str, lags: int = 4, n_splits: int = 2,
                  last_df_all: Optional[pd.DataFrame] = None):
    """
    Executes full forecasting pipeline:
    - Loads exchange rate data from database
    - Applies log transform and creates lag features
    - Forecasts next week's rate using ensemble model

    Parameters:
    - database_path: Path to SQLite DB
    - table_name: Table containing exchange rate
    - lags: Number of lag features to generate
    - n_splits: Number of splits for TimeSeriesSplit CV

    Returns:
    - fig: Forecast plot
    - metrics_df: RMSE, MAE, R-squared
    - y_next: Forecasted exchange rate (original scale)
    - last_df: DataFrame with last actual and predicted value with date
    """
    df = load_exchange_rate_from_db(database_path, table_name)
    df_lagged = log_and_create_lags(df, "exchange_rate", lags)

    y = df_lagged['log_rate']
    X = df_lagged.drop(columns=['log_rate'])

    fig, metrics_df, y_next, y_pred = forecast_next_week(X, y, n_splits=n_splits)

    # Prepare last actual vs predicted comparison
    last_index = df_lagged.index[-1]

    last_log_rate = df_lagged.loc[last_index, 'log_rate']
    last_log_rate_value = float(np.asarray(last_log_rate).item())
    last_actual = round(np.exp(last_log_rate_value), 4)

    # y_pred is already in original scale
    last_predicted = round(float(np.asarray(y_pred[-1]).item()), 4)
    last_df = pd.DataFrame({
        'Date': [last_index],
        'Last_Actual': [last_actual],
        'Last_Predicted': [last_predicted]
    }).set_index('Date')

    # Handle accumulation
    if last_df_all is None:
        last_df_all = last_df
    else:
        last_df_all = pd.concat([last_df_all, last_df])

    return fig, metrics_df, y_next, last_df_all
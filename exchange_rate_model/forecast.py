import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(1, input_shape)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def forecast_next_week(X, y, n_splits=2):
    """
    Train Lasso, LinearRegression, and LSTM as base models; LinearRegression as meta-model.
    Perform TimeSeriesSplit CV, plot prediction with interval, and forecast next week's rate.

    Parameters:
    - X: Feature matrix (DataFrame).
    - y: Target vector (Series, assumed to be log-transformed).
    - n_splits: Number of folds for TimeSeriesSplit.

    Returns:
    - fig: Plot of actual vs predicted and forecast.
    - metrics_df: DataFrame containing RMSE, MAE, R-squared.
    - y_next: Forecasted exchange rate for next week (in original scale).
    """
    from sklearn.base import clone

    set_seeds()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_y_test, all_y_pred = [], []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

        lasso = Lasso().fit(X_train, y_train)
        lr = LinearRegression().fit(X_train, y_train)
        lstm_model = build_lstm_model(X_train.shape[1])
        lstm_model.fit(X_train_reshaped, y_train.values, epochs=10, batch_size=1, verbose=0)

        base_preds = pd.DataFrame({
            'lasso': lasso.predict(X_test),
            'lr': lr.predict(X_test),
            'lstm': lstm_model.predict(X_test_reshaped).flatten()
        }, index=X_test.index)

        meta_model = LinearRegression().fit(base_preds, y_test)
        y_pred = meta_model.predict(base_preds)

        all_y_test.extend(np.exp(y_test))
        all_y_pred.extend(np.exp(y_pred))

    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)

    rmse = sqrt(mean_squared_error(all_y_test, all_y_pred))
    mae = mean_absolute_error(all_y_test, all_y_pred)
    r2 = r2_score(all_y_test, all_y_pred)
    metrics_df = pd.DataFrame({"RMSE": [rmse], "MAE": [mae], "R-squared": [r2]})

    # Refit on all data
    X_reshaped = X.values.reshape((X.shape[0], 1, X.shape[1]))
    lasso = Lasso().fit(X, y)
    lr = LinearRegression().fit(X, y)
    lstm_model = build_lstm_model(X.shape[1])
    lstm_model.fit(X_reshaped, y.values, epochs=10, batch_size=1, verbose=0)

    base_preds_all = pd.DataFrame({
        'lasso': lasso.predict(X),
        'lr': lr.predict(X),
        'lstm': lstm_model.predict(X_reshaped).flatten()
    }, index=X.index)

    meta_model = LinearRegression().fit(base_preds_all, y)

    # Forecast next week
    X_next = X.iloc[[-1]].copy()
    X_next_reshaped = X_next.values.reshape((1, 1, X.shape[1]))

    if isinstance(X.index[-1], pd.Timestamp):
        forecast_index = X.index[-1] + pd.Timedelta(weeks=1)
    else:
        forecast_index = X.index[-1] + 1

    base_next = pd.DataFrame({
        'lasso': lasso.predict(X_next),
        'lr': lr.predict(X_next),
        'lstm': lstm_model.predict(X_next_reshaped).flatten()
    }, index=[forecast_index])

    y_next_log = meta_model.predict(base_next)[0]
    y_next = np.exp(y_next_log)

    upper = all_y_pred + rmse
    lower = all_y_pred - rmse

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y.index, np.exp(y), label="Actual", color="black")
    ax.plot(y.index[-len(all_y_pred):], all_y_pred, label="Predicted", color="blue")
    ax.fill_between(y.index[-len(all_y_pred):], lower, upper, color="blue",
                    alpha=0.2, label="Prediction Interval")
    ax.scatter(forecast_index, y_next, label="Forecast (Next Week)",
               color="red", zorder=5)
    ax.axvline(forecast_index, color="gray", linestyle="--", alpha=0.5)

    ax.set_title("Actual vs Predicted Exchange Rate with Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exchange Rate")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return fig, metrics_df, y_next, all_y_pred
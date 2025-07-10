# Packages to be installed
import sqlite3
import pandas as pd

# Importing the data
def load_data_from_db(db_path: str, table_name: str, date_col: str = "date", rate_col: str = "Price") -> pd.DataFrame:
    """
    Load weekly NGN/USD exchange rate data from a remote SQLite .db file.

    Parameters:
        db_path (str): Path or URI to the .db SQLite database file.
        table_name (str): Name of the table containing the exchange rate data.
        date_col (str): Name of the date column in the table.
        rate_col (str): Name of the exchange rate column (usually 'price').

    Returns:
        pd.DataFrame: Cleaned dataframe with datetime index and one column named 'exchange_rate'.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Read the data into a DataFrame
    query = f"SELECT {date_col}, {rate_col} FROM {table_name}"
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    # Convert 'date' to datetime and sort
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)

    # Set datetime index
    df.set_index(date_col, inplace=True)

    # Rename column for consistency
    df.rename(columns={rate_col: "exchange_rate"}, inplace=True)

    return df

# Descripive analysis of exchange rate with plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, jarque_bera

sns.set_theme(style="whitegrid")

def summarize_with_plot(df: pd.DataFrame, column: str = "exchange_rate"):
    """
    Plot the exchange rate series using seaborn and matplotlib,
    with summary statistics displayed on the chart.
    The x-axis uses automatic date formatting to avoid clutter.

    Parameters:
        df (pd.DataFrame): DataFrame with datetime index and exchange rate column.
        column (str): The name of the exchange rate column (default is 'exchange_rate').

    Returns:
        None
    """
    # Drop missing values
    series = df[column].dropna()

    # Compute summary statistics
    desc = series.describe()
    jb_stat, jb_p = jarque_bera(series)

    stats_text = (
        f"Obs: {int(desc['count'])}\n"
        f"Mean: {desc['mean']:.2f}\n"
        f"Std: {desc['std']:.2f}\n"
        f"Skew: {skew(series):.2f}\n"
        f"Kurt: {kurtosis(series, fisher=True):.2f}\n"
        f"JB p-val: {jb_p:.4f}"
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y=column, ax=ax, color="royalblue", linewidth=2)

    ax.set_title("Weekly NGN/USD Exchange Rate", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Exchange Rate", fontsize=12)

    # Remove manual tick formatting
    ax.tick_params(axis='x', rotation=45)

    # Annotation box
    props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Creating log and lag of the exchage rate series
import numpy as np
import pandas as pd

def log_and_lag(dataframe: pd.DataFrame, column: str = "exchange_rate", n_lags: int = 3) -> pd.DataFrame:
    """
    Log-transform the original exchange rate column (overwrite it),
    and generate lag features from the logged series.

    Parameters:
        df (pd.DataFrame): Time-indexed DataFrame with exchange rate column.
        column (str): Name of the column to log-transform and lag.
        n_lags (int): Number of lagged features to generate.

    Returns:
        pd.DataFrame: Modified DataFrame with log-transformed column and lag features.
    """
    df = dataframe.copy()

    # Ensure the series is strictly positive before log-transforming
    if (df[column] <= 0).any():
        raise ValueError("Values must be strictly positive for log transformation.")

    # Overwrite the column with its log
    df[column] = np.log(df[column])

    # Add lag features
    for lag in range(1, n_lags + 1):
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)

    # Drop initial rows with NaN from lagging
    df.dropna(inplace=True)

    return df

# Plotting logged-lagged exchange rate series
def plot_logged_series(df):
    """
    Plot all log-transformed exchange rate series and lagged versions from the DataFrame.

    Assumes the DataFrame contains:
    - A datetime index
    - Log-transformed series in the main column
    - Lagged versions as additional columns (e.g., exchange_rate_lag_1, lag_2, etc.)

    Parameters:
        df (pd.DataFrame): DataFrame with datetime index and log/lags.

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot all numeric series in the dataframe
    for col in df.select_dtypes(include=["float", "int"]).columns:
        linestyle = '-' if 'lag' not in col else '--'
        sns.lineplot(x=df.index, y=df[col], ax=ax, label=col, linestyle=linestyle)

    ax.set_title("Log-Transformed NGN/USD Exchange Rate and Lagged Series", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("log(Exchange Rate)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()

# Splitting series into features and target then train and test
def time_series_split(dataframe, target_column: str = "exchange_rate", test_size: float = 0.2):
    """
    Split a time series DataFrame into train and test sets with feature/target separation.

    Parameters:
        df (pd.DataFrame): Cleaned, log-transformed, and lagged DataFrame.
        target_column (str): Column name to predict (e.g., 'exchange_rate' log-transformed).
        test_size (float): Fraction of the data to reserve for testing (0 < test_size < 1).

    Returns:
        X_train, X_test, y_train, y_test (chronologically split)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    # Feature matrix: all columns except the target
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    # Chronological split
    split_index = int(len(dataframe) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test

# Modelling
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reshape_for_lstm(features_train: pd.DataFrame) -> np.ndarray:
    """
    Reshape lagged features for LSTM input (samples, time steps, features).
    """
    return features_train.values.reshape((features_train.shape[0], 1, features_train.shape[1]))


def build_lstm_model(input_shape: tuple) -> Sequential:
    """
    Build a clean LSTM model using best TensorFlow practices.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def ensemble_lasso_linear_lstm(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    inverse_log: bool = True
) -> dict:
    """
    Trains Lasso (with scaling), Linear, and LSTM models and stacks them with a Linear meta-model.

    Parameters:
        features_train, target_train: Training data
        features_test, target_test: Test data
        inverse_log: If True, apply exponential to undo log-transform

    Returns:
        Dictionary of MAE, RMSE, R2, y_pred, and y_actual. Also plots predictions.
    """

    # --- Train Lasso with scaling ---
    lasso_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(alpha=0.01, max_iter=10000))
    ])
    lasso_model.fit(features_train, target_train)

    # --- Train Linear Regression (no scaling needed) ---
    linear_model = LinearRegression()
    linear_model.fit(features_train, target_train)

    # --- Prepare and train LSTM ---
    X_lstm_train = reshape_for_lstm(features_train)
    X_lstm_test = reshape_for_lstm(features_test)

    lstm_model = build_lstm_model(input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2]))
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    lstm_model.fit(
        X_lstm_train,
        target_train.values,
        epochs=50,
        batch_size=16,
        verbose=0,
        callbacks=[early_stop]
    )

    # --- Generate Base Predictions ---
    pred_lasso = lasso_model.predict(features_test)
    pred_linear = linear_model.predict(features_test)
    pred_lstm = lstm_model.predict(X_lstm_test, verbose=0).flatten()

    # --- Stack base predictions for meta-model ---
    meta_X = np.vstack([pred_lasso, pred_linear, pred_lstm]).T
    meta_model = LinearRegression()
    meta_model.fit(meta_X, target_test)

    final_preds = meta_model.predict(meta_X)

    # --- Optionally undo log transform ---
    if inverse_log:
        y_actual = np.exp(target_test)
        final_preds = np.exp(final_preds)
    else:
        y_actual = target_test

    # --- Evaluation Metrics ---
    mae = mean_absolute_error(y_actual, final_preds)
    rmse = np.sqrt(np.mean((y_actual - final_preds) ** 2))
    r2 = r2_score(y_actual, final_preds)

    # --- Plot Predictions ---
    y_actual = pd.Series(y_actual, index=target_test.index)
    final_preds = pd.Series(final_preds, index=target_test.index)

    plt.figure(figsize=(12, 5))
    plt.plot(y_actual, label="Actual", color="darkgreen", linewidth=2)
    plt.plot(final_preds, label="Ensemble Predicted", color="orange", linestyle="--", linewidth=2)
    plt.title("Actual vs Ensemble Predicted Exchange Rate", fontsize=14, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Exchange Rate (₦/$)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "y_pred": final_preds,
        "y_actual": y_actual
    }


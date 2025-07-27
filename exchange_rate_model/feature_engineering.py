import pandas as pd
import numpy as np

def log_and_create_lags(df: pd.DataFrame, column: str = "exchange_rate", n_lags: int = 4) -> pd.DataFrame:
    """
    Log-transform a specified column and create lagged features from it.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with exchange rate data.
        column (str): Column name to transform.
        n_lags (int): Number of lag features to create.
        
    Returns:
        pd.DataFrame: DataFrame with log-transformed and lag features.
    """
    df = df.copy()
    
    # Log transformation
    df["log_rate"] = np.log(df[column])
    
    # Create lag features
    for i in range(1, n_lags + 1):
        df[f"log_rate_lag{i}"] = df["log_rate"].shift(i)

    # Drop the original series column explicitly
    if column in df.columns:
        df.drop(columns=[column], inplace=True)
    
    # Drop rows with NaNs due to shifting
    df.dropna(inplace=True)

    return df

def train_test_split_time_series(df: pd.DataFrame, target_col: str = "log_rate",
                                 test_size: float = 0.2):
    """
    Time-aware train-test split for time series.

    Args:
        df (pd.DataFrame): DataFrame with features and target.
        target_col (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    df = df.copy()
    split_point = int(len(df) * (1 - test_size))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    return X_train, X_test, y_train, y_test
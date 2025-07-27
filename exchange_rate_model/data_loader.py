import sqlite3
import pandas as pd
from pathlib import Path

def load_exchange_rate_from_db(db_path: str, table_name: str = "ngn_usd_data") -> pd.DataFrame:
    """
    Load exchange rate data from a SQLite database.

    Args:
        db_path (str): Path to the SQLite .db file.
        table_name (str): Name of the table to query. Default is 'ngn_usd_data'.

    Returns:
        pd.DataFrame: DataFrame with datetime index and exchange_rate column.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()

    # Ensure datetime format and sorting
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    # Clean column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    if 'exchange_rate' not in df.columns:
        raise ValueError("Expected 'exchange_rate' column not found in the table.")

    df = df[['exchange_rate']].dropna()

    return df
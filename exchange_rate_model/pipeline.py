import joblib
import pandas as pd
from exchange_rate_model.module import ensemble_lasso_linear_lstm

# Load saved models or retrain dynamically if needed
def predict_next_exchange_rate(new_data: pd.DataFrame) -> float:
    """
    Takes in most recent logged and lagged data (1 row), returns next week's predicted exchange rate.

    Parameters:
        new_data (pd.DataFrame): A single-row DataFrame with same structure as training features.

    Returns:
        float: Predicted exchange rate (₦/$) on original scale.
    """

    # Load training artifacts – for now, we simulate with the same model pipeline
    # In production, you’d load the saved models
    # model = joblib.load("model/ensemble_model.joblib")

    # Placeholder: Use dummy 1-row input as both train and test to simulate response
    pred = ensemble_lasso_linear_lstm(
        features_train=new_data,
        target_train=pd.Series([new_data.iloc[0, 0]]),  # dummy single target
        features_test=new_data,
        target_test=pd.Series([new_data.iloc[0, 0]]),
        inverse_log=True
    )

    return float(pred["y_pred"].values[0])

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Helper Functions ---

def reshape_for_lstm(features: pd.DataFrame) -> np.ndarray:
    return features.to_numpy().reshape((features.shape[0], 1, features.shape[1]))

def train_lasso_regression(X_train: pd.DataFrame, y_train: pd.Series):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(alpha=0.01, max_iter=10000))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def compile_and_train(model, X, y):
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=[early_stop])

def train_lstm_model(X_train: pd.DataFrame, y_train: pd.Series):
    tf.keras.backend.clear_session()
    X_lstm = reshape_for_lstm(X_train)
    model = Sequential([
        Input(shape=(X_lstm.shape[1], X_lstm.shape[2])),
        LSTM(50, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    compile_and_train(model, X_lstm, y_train.to_numpy())
    return model

# --- Cross-Validated Meta-Model Trainer with best fold selection ---

def train_ensemble_meta_model_cv(X: pd.DataFrame, y: pd.Series):
    """
    Perform cross-validation on ensemble model to find best number of splits and return metrics and plot.

    Parameters:
        X (pd.DataFrame): Features (log scale)
        y (pd.Series): Target (log scale)

    Returns:
        pd.DataFrame: Evaluation metrics
    """
    best_rmse = float('inf')
    best_metrics = None
    best_y_true = None
    best_y_pred = None
    best_rmse_val = None

    for n_splits in range(2, 3):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        fold_preds = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            lasso = train_lasso_regression(X_train, y_train)
            linear = train_linear_regression(X_train, y_train)
            lstm = train_lstm_model(X_train, y_train)

            meta_X_train = np.vstack([
                lasso.predict(X_train),
                linear.predict(X_train),
                lstm.predict(reshape_for_lstm(X_train), verbose=0).flatten()
            ]).T

            meta_model = LinearRegression()
            meta_model.fit(meta_X_train, y_train)

            meta_X_test = np.vstack([
                lasso.predict(X_test),
                linear.predict(X_test),
                lstm.predict(reshape_for_lstm(X_test), verbose=0).flatten()
            ]).T

            y_pred_log = meta_model.predict(meta_X_test)
            y_pred = pd.Series(np.exp(y_pred_log), index=y_test.index)
            y_true = pd.Series(np.exp(y_test), index=y_test.index)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            fold_metrics.append({'MAE': mae, 'RMSE': rmse, 'R_squared': r2})
            fold_preds.append((y_true, y_pred, rmse))

        avg_rmse = np.mean([m['RMSE'] for m in fold_metrics])
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_metrics = pd.DataFrame(fold_metrics).mean().to_frame().T.round(4)
            best_y_true = pd.Series(fold_preds[-1][0].values, index=fold_preds[-1][0].index)
            best_y_pred = pd.Series(fold_preds[-1][1].values, index=fold_preds[-1][0].index)
            best_rmse_val = fold_preds[-1][2]

    # Defensive check
    if best_y_true is None or best_y_pred is None:
        raise ValueError("Best predictions or actuals are not defined. Check fold predictions.")

    # Final plot
    plt.figure(figsize=(10, 4))
    plt.plot(best_y_true.index.to_numpy(), best_y_true.to_numpy(), label='Actual', color='black')
    plt.plot(best_y_pred.index.to_numpy(), best_y_pred.to_numpy(), label='Predicted', color='blue')
    plt.fill_between(best_y_pred.index.to_numpy(), best_y_pred.to_numpy() - best_rmse_val, best_y_pred.to_numpy() + best_rmse_val, color='blue', alpha=0.2, label='±1 RMSE')
    plt.title('Actual vs Predicted (Best Fold) with RMSE Boundary')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_metrics
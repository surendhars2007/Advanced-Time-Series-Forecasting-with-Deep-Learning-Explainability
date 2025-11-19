#!/usr/bin/env python3
"""
advanced_time_series_forecasting.py

End-to-end example:
- Synthetic multivariate time series generation
- LSTM forecasting model with MC Dropout for uncertainty
- Optuna hyperparameter tuning
- Evaluation: RMSE, MAE, Coverage, Interval Width
- Plots: Actual vs Predicted (with intervals), residuals, histogram, interval width, coverage indicator

Dependencies:
pip install numpy pandas scikit-learn matplotlib optuna tensorflow

Run:
python advanced_time_series_forecasting.py
"""

import os
import math
import random
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Synthetic data generator
# -------------------------
def generate_synthetic_data(n=3000, seasonal_period=50, seed=SEED) -> pd.DataFrame:
    """
    Generates a single-series synthetic time series with trend, seasonality and noise.
    Returns a pandas DataFrame with a single column 'value'.
    """
    rng = np.arange(n)
    np.random.seed(seed)
    trend = 0.01 * rng
    seasonal = 2.0 * np.sin(2 * np.pi * rng / seasonal_period) + 0.5 * np.sin(2 * np.pi * rng / (seasonal_period*7))
    noise = np.random.normal(scale=0.5, size=n)
    series = trend + seasonal + noise
    df = pd.DataFrame({"value": series})
    return df

# -------------------------
# Sequence creation
# -------------------------
def create_dataset(series: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create X (seq_len windows) and y (next point) arrays from a 1D series.
    """
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    X = np.array(X)  # shape (n_samples, seq_len, 1)
    y = np.array(y)  # shape (n_samples, 1)
    return X, y

# -------------------------
# Model builder
# -------------------------
def build_lstm_model(seq_len: int, n_units: int = 64, dropout_rate: float = 0.2, lr: float = 1e-3) -> tf.keras.Model:
    """
    Build a simple LSTM with dropout (dropout used for MC Dropout at inference).
    """
    model = Sequential([
        LSTM(n_units, input_shape=(seq_len, 1), return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

# -------------------------
# MC Dropout prediction helper
# -------------------------
def mc_dropout_predict(model: tf.keras.Model, X: np.ndarray, T: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run T stochastic forward passes with dropout enabled (training=True).
    Returns (mean_pred, std_pred) with shapes (n_samples, 1).
    """
    preds = []
    for _ in range(T):
        preds.append(model(X, training=True).numpy())
    preds = np.stack(preds, axis=0)  # shape (T, n_samples, 1)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

# -------------------------
# Optuna objective
# -------------------------
def optuna_objective(trial, X_train, y_train, X_val, y_val, seq_len):
    n_units = trial.suggest_int("n_units", 32, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = build_lstm_model(seq_len, n_units=n_units, dropout_rate=dropout_rate, lr=lr)
    # short training for trial
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=6, batch_size=batch_size, verbose=0)
    preds = model.predict(X_val)
    val_mse = mean_squared_error(y_val, preds)
    return val_mse

# -------------------------
# Plot utilities
# -------------------------
def save_or_show_plot(savepath: str):
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# -------------------------
# Main pipeline
# -------------------------
def main(args):
    # 1. Generate data
    df = generate_synthetic_data(n=args.n_points, seasonal_period=args.seasonal_period, seed=SEED)
    series = df['value'].values.reshape(-1, 1)

    # 2. Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    # 3. Create sequences
    seq_len = args.seq_len
    X_all, y_all = create_dataset(scaled.flatten(), seq_len)
    # reshape to (n_samples, seq_len, 1)
    X_all = X_all.reshape((X_all.shape[0], seq_len, 1))
    y_all = y_all.reshape((-1, 1))

    # Train/val/test split
    n_total = len(X_all)
    n_train = int((1.0 - args.val_ratio - args.test_ratio) * n_total)
    n_val = int(args.val_ratio * n_total)
    X_train = X_all[:n_train]; y_train = y_all[:n_train]
    X_val = X_all[n_train:n_train+n_val]; y_val = y_all[n_train:n_train+n_val]
    X_test = X_all[n_train+n_val:]; y_test = y_all[n_train+n_val:]

    print(f"Samples: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # 4. Optuna HPO (optional)
    best_params = None
    if args.do_optuna:
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
        func = lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, seq_len)
        study.optimize(func, n_trials=args.optuna_trials, show_progress_bar=True)
        best_params = study.best_params
        print("Optuna best params:", best_params)
    else:
        # default
        best_params = {'n_units': 64, 'dropout_rate': 0.2, 'lr': 1e-3, 'batch_size': 32}

    # 5. Build final model & train
    n_units = int(best_params.get('n_units', 64))
    dropout_rate = float(best_params.get('dropout_rate', 0.2))
    lr = float(best_params.get('lr', 1e-3))
    batch_size = int(best_params.get('batch_size', 32))

    model = build_lstm_model(seq_len, n_units=n_units, dropout_rate=dropout_rate, lr=lr)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=batch_size, verbose=1)

    # 6. MC Dropout predictions
    mean_pred_s, std_pred_s = mc_dropout_predict(model, X_test, T=args.mc_T)

    # 7. Inverse scale
    mean_pred = scaler.inverse_transform(mean_pred_s)
    std_pred = std_pred_s * scaler.scale_[0]  # approximate unscale for std
    y_test_orig = scaler.inverse_transform(y_test)

    # 8. Compute intervals (80% and 95%)
    z80 = 1.28155
    z95 = 1.95996
    lower80 = mean_pred - z80 * std_pred
    upper80 = mean_pred + z80 * std_pred
    lower95 = mean_pred - z95 * std_pred
    upper95 = mean_pred + z95 * std_pred

    # 9. Metrics
    rmse = math.sqrt(mean_squared_error(y_test_orig, mean_pred))
    mae = mean_absolute_error(y_test_orig, mean_pred)

    # coverage
    coverage80 = np.mean((y_test_orig >= lower80) & (y_test_orig <= upper80))
    coverage95 = np.mean((y_test_orig >= lower95) & (y_test_orig <= upper95))

    avg_width80 = np.mean(upper80 - lower80)
    avg_width95 = np.mean(upper95 - lower95)

    print("\n===== METRICS =====")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"80% Coverage: {coverage80:.4f}, AvgWidth: {avg_width80:.6f}")
    print(f"95% Coverage: {coverage95:.4f}, AvgWidth: {avg_width95:.6f}")

    # 10. Summary table (pandas)
    summary = pd.DataFrame([
        {'Metric': 'RMSE', '80%': rmse, '95%': rmse},
        {'Metric': 'MAE', '80%': mae, '95%': mae},
        {'Metric': 'Coverage', '80%': coverage80, '95%': coverage95},
        {'Metric': 'AvgWidth', '80%': avg_width80, '95%': avg_width95},
    ])
    print("\nSummary Table:\n", summary.to_string(index=False))

    # 11. PLOTS
    # create output dir
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # convenience variables
    n_points = len(mean_pred)
    xaxis = np.arange(n_points)

    # Actual vs Predicted with 95% PI
    plt.figure(figsize=(12, 5))
    plt.plot(xaxis, y_test_orig.flatten(), label='Actual', linewidth=1.5)
    plt.plot(xaxis, mean_pred.flatten(), label='Predicted Mean', linewidth=1.5)
    plt.fill_between(xaxis, lower95.flatten(), upper95.flatten(), alpha=0.25, label='95% PI')
    plt.fill_between(xaxis, lower80.flatten(), upper80.flatten(), alpha=0.15, label='80% PI')
    plt.title('Actual vs Predicted with Prediction Intervals')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    save_or_show_plot(os.path.join(args.output_dir, 'actual_vs_predicted.png') if args.output_dir else None)

    # Residuals
    residuals = y_test_orig.flatten() - mean_pred.flatten()
    plt.figure(figsize=(12,4))
    plt.plot(xaxis, residuals, label='Residuals')
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Residuals Time Series')
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.grid(True)
    save_or_show_plot(os.path.join(args.output_dir, 'residuals.png') if args.output_dir else None)

    # Residual histogram
    plt.figure(figsize=(8,4))
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    save_or_show_plot(os.path.join(args.output_dir, 'residuals_hist.png') if args.output_dir else None)

    # Interval width plot (80% and 95%)
    width80 = (upper80 - lower80).flatten()
    width95 = (upper95 - lower95).flatten()
    plt.figure(figsize=(12,4))
    plt.plot(xaxis, width80, label='80% Interval Width')
    plt.plot(xaxis, width95, label='95% Interval Width', alpha=0.8)
    plt.title('Prediction Interval Width Over Time')
    plt.xlabel('Sample')
    plt.ylabel('Interval Width')
    plt.legend()
    plt.grid(True)
    save_or_show_plot(os.path.join(args.output_dir, 'interval_width.png') if args.output_dir else None)

    # Coverage indicator (95% shown)
    covered95 = ((y_test_orig >= lower95) & (y_test_orig <= upper95)).astype(int).flatten()
    plt.figure(figsize=(12,3))
    plt.plot(xaxis, covered95, drawstyle='steps-mid')
    plt.title('Coverage Indicator (1 = covered by 95% PI, 0 = missed)')
    plt.xlabel('Sample')
    plt.ylabel('Covered?')
    plt.grid(True)
    save_or_show_plot(os.path.join(args.output_dir, 'coverage_indicator.png') if args.output_dir else None)

    print("\nPlots saved to:", args.output_dir if args.output_dir else "shown interactively")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Time Series Forecasting script')
    parser.add_argument('--n_points', type=int, default=3000, help='number of synthetic data points')
    parser.add_argument('--seasonal_period', type=int, default=50, help='seasonal period for synthetic data')
    parser.add_argument('--seq_len', type=int, default=50, help='input sequence length')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test ratio')
    parser.add_argument('--do_optuna', action='store_true', help='run optuna hyperparameter tuning')
    parser.add_argument('--optuna_trials', type=int, default=10, help='number of optuna trials')
    parser.add_argument('--epochs', type=int, default=20, help='training epochs for final model')
    parser.add_argument('--mc_T', type=int, default=50, help='MC dropout forward passes')
    parser.add_argument('--output_dir', type=str, default='', help='directory to save plots (if empty, show interactively)')
    args = parser.parse_args()

    main(args)

import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from train_predictive_model import aggregate_all_coins_data, prepare_data, train_and_evaluate_all
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/coin_mapping.json"))
TRAINER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trainer"))
os.makedirs(TRAINER_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(TRAINER_DIR, "latest_model.pkl")
TRAIN_STATS_PATH = os.path.join(TRAINER_DIR, "train_stats.json")

TRADING_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trading_data"))
os.makedirs(TRADING_DATA_DIR, exist_ok=True)
SIGNALS_PATH = os.path.join(TRADING_DATA_DIR, "latest_signals.json")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))

MIN_ROWS_TO_TRAIN = 200  # Set threshold
CHECK_INTERVAL = 120  # seconds

def check_stationarity(series, cutoff=0.05):
    """Returns True if series is stationary, False otherwise."""
    result = adfuller(series.dropna())
    return result[1] < cutoff  # p-value < cutoff means stationary

def difference_series(df, col='close'):
    df = df.copy()
    df[col] = df[col].diff()
    df = df.dropna()
    return df

def train_once():
    all_df = aggregate_all_coins_data(MAPPING_PATH)
    status_msgs = []
    status_msgs.append(f"Data shape: {all_df.shape}")
    if len(all_df) >= MIN_ROWS_TO_TRAIN:
        # --- Stationarity & Seasonality Check ---
        stationary = check_stationarity(all_df['close'])
        if not stationary:
            status_msgs.append("Close price is non-stationary. Differencing applied.")
            all_df = difference_series(all_df, 'close')
        else:
            status_msgs.append("Close price is stationary.")

        # Optional: Seasonality check (for logging)
        try:
            decomposition = seasonal_decompose(all_df['close'], model='additive', period=24)
            status_msgs.append("Seasonality detected. See decomposition plot in logs.")
            decomposition.plot()
            import matplotlib.pyplot as plt
            plt.show()
        except Exception as e:
            status_msgs.append(f"Seasonality check skipped: {e}")

        X, y, closes = prepare_data(all_df)
        tscv = TimeSeriesSplit(n_splits=5)
        from train_predictive_model import get_all_classifiers
        classifiers = get_all_classifiers()

        test_results = {}

        for name, clf in classifiers.items():
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                # Log class imbalance
                print(f"Train class distribution: {y_train.value_counts().to_dict()}")
                print(f"Test class distribution: {y_test.value_counts().to_dict()}")
                # Downsample majority class for training
                X_train_ds, y_train_ds = downsample(X_train, y_train)
                clf.fit(X_train_ds, y_train_ds)
                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

                # Test set metrics
                test_results[name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0)
                }

        stats = {
            "timestamp": time.time(),
            "rows_trained": len(all_df),
            "test_results": test_results
        }
        with open(TRAIN_STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2)
        status_msgs.append("Test set metrics for all models saved.")
    else:
        status_msgs.append(f"Not enough data to train ({len(all_df)}/{MIN_ROWS_TO_TRAIN})")
    return status_msgs

def get_latest_data_mtime(data_dir):
    mtimes = []
    for fname in os.listdir(data_dir):
        if fname.endswith("_model_data.csv"):
            mtimes.append(os.path.getmtime(os.path.join(data_dir, fname)))
    return max(mtimes) if mtimes else 0

def continuous_train_loop():
    print("Starting continuous training loop...")
    last_train_time = 0
    while True:
        latest_data_time = get_latest_data_mtime(DATA_DIR)
        if latest_data_time > last_train_time:
            train_once()
            last_train_time = time.time()
        else:
            print("No new data. Skipping retrain.")
        time.sleep(CHECK_INTERVAL)

def load_recent_coin_data(symbol, data_dir, hours=24):
    path = os.path.join(data_dir, f"{symbol}_model_data.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(hours=hours)
        df = df[df["timestamp"] >= cutoff]
    return df

def aggregate_all_coins_data(mapping_path):
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    dfs = []
    for symbol in mapping.keys():
        df = load_recent_coin_data(symbol, DATA_DIR)
        if not df.empty:
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return all_df

if __name__ == "__main__":
    continuous_train_loop()
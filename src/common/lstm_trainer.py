import os
import glob
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIG ---
DATA_DIR = "data/scraped_data"
MODEL_PATH = "data/models"
MIN_ROWS = 50
MIN_GOOD_COINS = 12
MODEL_OUT_PATH = os.path.join(MODEL_PATH, "trained_lstm_model.h5")
REQUIRED_FEATURES = ["timestamp", "open", "high", "low", "close", "volume", "rsi", "macd"]
REQUIRED_COLS = ["timestamp", "close", "target"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LSTMTrainer")

def scan_csvs_for_training(data_dir, min_rows=50, min_good_coins=12):
    files = glob.glob(os.path.join(data_dir, "*_model_data.csv"))
    good_files = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if all(col in df.columns for col in REQUIRED_FEATURES) and len(df.dropna(subset=REQUIRED_FEATURES)) >= min_rows:
                good_files.append(f)
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
    logger.info(f"Found {len(good_files)} coins with at least {min_rows} valid rows.")
    return good_files

def load_and_prepare_data(good_files):
    all_data = []
    for f in good_files:
        try:
            df = pd.read_csv(f)
            df["symbol"] = os.path.basename(f).split("_")[0]
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f} due to error: {e}")
    if not all_data:
        raise ValueError("No valid data files found.")
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def preprocess_data(df, sequence_length=20):
    df = df.copy()
    df = df[pd.to_datetime(df["timestamp"], errors="coerce").notnull()]
    df = df.sort_values(by=["symbol", "timestamp"])
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=REQUIRED_FEATURES + ["target"])
    df['symbol_id'] = df['symbol'].astype("category").cat.codes

    # Standardize features
    features = ["open", "high", "low", "close", "volume", "rsi", "macd"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Prepare sequences for LSTM
    X, y = [], []
    symbols = df['symbol'].unique()
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol]
        values = symbol_df[features].values
        targets = symbol_df["target"].values
        for i in range(len(values) - sequence_length):
            X.append(values[i:i+sequence_length])
            y.append(targets[i+sequence_length])
    X = np.array(X)
    y = np.array(y).astype(int)
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    logger.info("Starting LSTM training pipeline...")

    good_files = scan_csvs_for_training(DATA_DIR, min_rows=MIN_ROWS, min_good_coins=MIN_GOOD_COINS)
    if len(good_files) < MIN_GOOD_COINS:
        logger.error(f"Not enough coins with sufficient data ({len(good_files)}/{MIN_GOOD_COINS}). Scrape more data and try again.")
        return

    df = load_and_prepare_data(good_files)
    X, y, scaler = preprocess_data(df, sequence_length=20)
    logger.info(f"Training LSTM on {X.shape[0]} sequences.")

    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    model.save(MODEL_OUT_PATH)
    logger.info(f"LSTM model saved to {MODEL_OUT_PATH}")

    # Save scaler for later use
    scaler_path = MODEL_OUT_PATH.replace(".h5", "_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting LSTM training script gracefully.")
        import sys
        sys.exit(0)
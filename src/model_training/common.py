import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import ta
from src.trading_bot.technical_indicators import TechnicalIndicators

logger = logging.getLogger("Common")

REQUIRED_FEATURES = ["timestamp", "open", "high", "low", "close", "volume", "rsi", "macd"]

def preprocess_data(df):
    df = df.copy()
    df = df[pd.to_datetime(df["timestamp"], errors="coerce").notnull()]
    df["symbol_id"] = df["symbol"].astype("category").cat.codes
    df = df.sort_values(by=["symbol", "timestamp"])
    df = df.dropna(axis=1, how="all")

    # Use the new technical indicators module for all calculations
    try:
        indicators = TechnicalIndicators()
        df = indicators.calculate_all_indicators(df)
        
        # Additional derived features
        df["volatility"] = df["close"].pct_change().rolling(window=10).std()
        df["parabolic_sar"] = ta.trend.PSARIndicator(df["high"], df["low"], df["close"]).psar()
        
    except Exception as e:
        logger.warning(f"Error calculating technical indicators, using fallbacks: {e}")
        # Fallback calculations if the module fails
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["macd"] = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9).macd()
        df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
        df["volatility"] = df["close"].pct_change().rolling(window=10).std()

    # Add time-based features
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Add returns and lag features
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    for i in range(1, 4):
        df[f"return_lag{i}"] = df["return"].shift(i)

    # Add rolling features
    df["rsi_rolling_mean"] = df["rsi"].rolling(3).mean()
    df["close_rolling_std"] = df["close"].rolling(5).std()

    # Log initial DataFrame shape and columns
    logger.info(f"🔍 Starting preprocessing: shape = {df.shape}, columns = {df.columns.tolist()}")

    # Remove garbage rows
    df = df[df["volume"] > 0]

    # Comment out the early dropna
    # df.dropna(inplace=True)

    # Add smarter labels
    df["future_return"] = df["close"].shift(-3) / df["close"] - 1

    # Check for all NaN future returns
    if df["future_return"].isna().all():
        logger.warning("⚠️ All future returns are NaN, skipping.")
        return pd.DataFrame(), pd.Series(dtype=int), np.array([])

    # Create 'target' column with safe handling of NaN values
    df["target"] = pd.cut(
        df["future_return"],
        bins=[-float("inf"), -0.003, 0.003, float("inf")],
        labels=[-1, 0, 1]
    )

    # Log target class distribution
    logger.info(f"📊 Target class distribution:\n{df['target'].value_counts()}")

    # Drop rows with NaN in 'target'
    df.dropna(subset=["target"], inplace=True)

    # Replace infinity values with NaN in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["target", "symbol_id"])
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Impute missing values in numeric columns with their mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Ensure scaled values are explicitly cast to float32 before assignment
    for symbol in df["symbol"].unique():
        mask = df["symbol"] == symbol
        if df.loc[mask].shape[0] < 5:
            logger.warning(f"⚠️ Symbol {symbol} has too few rows to scale, skipping.")
            continue
        scaled_values = StandardScaler().fit_transform(df.loc[mask, numeric_cols].astype(float)).astype(np.float32)
        df.loc[mask, numeric_cols] = scaled_values    # Check if DataFrame is empty before returning
    if df.empty:
        logger.warning("❌ DataFrame empty after preprocessing.")
        return pd.DataFrame(), pd.Series(dtype=int), np.array([])

    logger.info(f"✅ Preprocessing done: shape = {df.shape}, target dist =\n{df['target'].value_counts()}")

    # Exclude non-feature columns including file_version_id
    exclude_cols = ["timestamp", "target", "symbol", "file_version_id"]
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df["target"].astype(int)
    groups = df["symbol_id"].values

    drop_cols = [col for col in X.columns if X[col].nunique() <= 1]
    X.drop(columns=drop_cols, inplace=True)

    return X, y, groups

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✅ Model saved to {path}")

def fill_missing_candles(df, interval='1min'):
    all_filled = []
    for symbol in df["symbol"].unique():
        sub = df[df["symbol"] == symbol].copy()
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
        sub = sub.set_index("timestamp").sort_index()
        sub = sub.resample(interval).asfreq()
        sub[["open", "high", "low", "close"]] = sub[["open", "high", "low", "close"]].ffill()
        if "volume" in sub.columns:
            sub["volume"] = sub["volume"].fillna(0)
        sub["symbol"] = symbol  # Restore symbol column
        all_filled.append(sub.reset_index())
    return pd.concat(all_filled, ignore_index=True)
import os
import glob
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import subprocess

# --- CONFIG ---
DATA_DIR = "data/scraped_data"
MODEL_PATH = "data/models"
MIN_ROWS = 50
MIN_GOOD_COINS = 12  # Set your threshold here
MODEL_OUT_PATH = os.path.join(MODEL_PATH, "trained_model.pkl")
REQUIRED_FEATURES = ["timestamp", "open", "high", "low", "close", "volume", "rsi", "macd"]
REQUIRED_COLS = ["timestamp", "close", "target"]
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "min_samples_split": 8,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "oob_score": True,  # Use OOB for validation
    "bootstrap": True,  # Enable bootstrap sampling
    
}

# Most Stable Base Model Tune
# RF_PARAMS = {
#     "n_estimators": 200,
#     "max_depth": 12,
#     "min_samples_leaf": 4,
#     "min_samples_split": 8,
#     "max_features": "sqrt",
#     "class_weight": "balanced",
#     "random_state": 42,
#     "n_jobs": -1,
#     "oob_score": True,  
#     "bootstrap": True,  
    
# }

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Trainer")

def scan_parquets_for_training(data_dir, min_rows=50, min_good_coins=12):
    files = glob.glob(os.path.join(data_dir, "*_model_data.parquet"))
    good_files = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
            if missing_cols:
                logger.warning(f"{f} missing columns: {missing_cols}")
                continue
            # Only count rows where all required features are present
            n_valid = len(df.dropna(subset=["rsi", "macd"]))
            if n_valid < min_rows:
                logger.warning(f"{f} only {n_valid} valid rows (need {min_rows})")
                continue
            good_files.append(f)
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
    logger.info(f"Found {len(good_files)} coins with at least {min_rows} valid rows.")
    return good_files

def validate_and_rescrape_data(data_dir, required_features, min_rows=50):
    for fname in os.listdir(data_dir):
        if not fname.endswith("_model_data.parquet"):
            continue
        fpath = os.path.join(data_dir, fname)
        symbol = fname.replace("_model_data.parquet", "")
        try:
            df = pd.read_parquet(fpath)
            missing = [col for col in required_features if col not in df.columns or df[col].isnull().sum() > 0]
            if len(df) < min_rows or missing:
                logger.warning(f"{symbol}: Rescraping due to missing/NaN columns: {missing} or too few rows ({len(df)})")
                subprocess.run(["python", "src/common/data_scraper.py", symbol], check=True)
        except Exception as e:
            logger.warning(f"{symbol}: Error reading Parquet ({e}), rescraping...")
            subprocess.run(["python", "src/common/data_scraper.py", symbol], check=True)

def load_and_prepare_data(good_files):
    all_data = []
    for f in good_files:
        try:
            df = pd.read_parquet(f)
            df["symbol"] = os.path.basename(f).split("_")[0]
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f} due to error: {e}")
    if not all_data:
        raise ValueError("No valid data files found.")
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def preprocess_data(df, drop_low_importance=False, importance_threshold=0.01):
    df = df.copy()

    # Ensure valid timestamps
    df = df[pd.to_datetime(df["timestamp"], errors="coerce").notnull()]
    df["symbol_id"] = df["symbol"].astype("category").cat.codes
    df = df.sort_values(by=["symbol", "timestamp"])
    df = df.dropna(axis=1, how="all")

    # Print missing counts for required features
    print("Missing value counts for required features:")
    print(df[REQUIRED_FEATURES].isnull().sum())

    # Volume sanity check before scaling
    print("Missing values before scaling:")
    print(df[["symbol", "volume"]].isnull().groupby(df["symbol"]).sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["target", "symbol_id"])

    # Drop rows with NaNs in numeric columns
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    logger.info(f"Dropped {before - len(df)} rows with NaNs in numeric features before scaling.")

    # Standard scale per-symbol
    for symbol in df["symbol"].unique():
        mask = df["symbol"] == symbol
        df.loc[mask, numeric_cols] = StandardScaler().fit_transform(df.loc[mask, numeric_cols])

        # Post-scaling audits
        if df.loc[mask, "volume"].isnull().any():
            logger.warning(f"{symbol}: volume has NaNs after scaling")
        if (df.loc[mask, "volume"] == 0).sum() > 0:
            logger.warning(f"{symbol}: volume has 0s after scaling")

    # Final filtering
    if len(df) == 0:
        raise ValueError("No rows left after dropping those with missing required features.")
    if "target" not in df.columns:
        raise ValueError("No 'target' column found in data. Please ensure targets are generated.")
    df = df[df["target"].notna()]

    # Prepare features/labels
    X = df.drop(columns=["timestamp", "target", "symbol"])
    y = df["target"].astype(int)
    symbol_id = df["symbol_id"].values

    # Convert to DataFrame to retain column names
    col_names = X.columns
    X = pd.DataFrame(np.nan_to_num(X), columns=col_names)

    if drop_low_importance:
        logger.info("Running feature importance filter...")
        if len(X) == 0:
            raise ValueError("No data available for feature importance filtering.")

        temp_model = RandomForestClassifier(**RF_PARAMS)
        temp_model.fit(X, y)

        importances = temp_model.feature_importances_
        feat_imp = pd.Series(importances, index=col_names)
        important_cols = feat_imp[feat_imp >= importance_threshold].index

        logger.info(f"Keeping {len(important_cols)} important features out of {len(feat_imp)}")

        important_features_path = os.path.join(MODEL_PATH, "important_features.json")
        with open(important_features_path, 'w') as f:
            json.dump(list(important_cols), f, indent=2)

        logger.info(f"Saved important features to {important_features_path}")
        X = X[important_cols]

    return X, y.values, symbol_id


def train_model(X, y, groups):
    unique_groups = np.unique(groups)
    n_train = int(0.8 * len(unique_groups))
    train_groups = unique_groups[:n_train]
    test_groups = unique_groups[n_train:]
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Final Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}")
    return model, (mse, mae)

def save_model(model, scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")

def save_scaler(scaler, path):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def add_technical_indicators(df):
    df = df.copy()
    import talib
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)
    macd, _, _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd
    # Remove warmup NaNs
    #df = df.dropna(subset=["rsi", "macd"])
    return df

# --- Merge Refinement Data ---
def merge_refinement_data(refinement_csv, main_csv, important_features):
    try:
        ref_df = pd.read_csv(refinement_csv)
        main_df = pd.read_csv(main_csv)

        # Keep only the required features and outcome
        required_cols = important_features + ['outcome']
        ref_df_cleaned = ref_df[required_cols].copy()

        # Drop NaNs or rows with missing outcome
        ref_df_cleaned.dropna(subset=required_cols, inplace=True)

        # Merge into main training data
        combined = pd.concat([main_df, ref_df_cleaned], ignore_index=True)
        combined.drop_duplicates(inplace=True)

        combined.to_csv(main_csv, index=False)
        logger.info(f"[+] Successfully merged {len(ref_df_cleaned)} rows into training data.")
        return combined
    except Exception as e:
        logger.error(f"[!] Failed to merge refinement data: {e}")
        return None

# --- Retrain Model ---
def retrain_model(updated_df, important_features):
    try:
        X = updated_df[important_features]
        y = updated_df["outcome"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)

        # Save the updated model
        save_model(model, None, MODEL_OUT_PATH)
        logger.info("[+] Model retrained and saved successfully.")
    except Exception as e:
        logger.error(f"[!] Failed to retrain model: {e}")

# --- Main Incremental Learning Logic ---
def refine_model_from_logs():
    try:
        refinement_path = "logs/refinement_data.csv"
        main_training_path = "data/training_data.csv"

        # Load important_features list
        with open(os.path.join(MODEL_PATH, "important_features.json"), "r") as f:
            important_features = json.load(f)

        # Merge and retrain
        updated_df = merge_refinement_data(refinement_path, main_training_path, important_features)
        if updated_df is not None:
            retrain_model(updated_df, important_features)
    except Exception as e:
        logger.error(f"[!] Failed to refine model from logs: {e}")

def main():
    global REQUIRED_FEATURES
    logger.info("Starting training pipeline...")

    # 1. Scan for all Parquet files (do NOT filter by valid rows yet)
    files = glob.glob(os.path.join(DATA_DIR, "*_model_data.parquet"))
    if not files:
        raise ValueError("No data files found in scraped_data directory.")

    # 2. Load and concatenate all data
    all_data = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["symbol"] = os.path.basename(f).split("_")[0]
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f} due to error: {e}")
    if not all_data:
        raise ValueError("No valid data files found.")
    combined_df = pd.concat(all_data, ignore_index=True)

    # --- Volume Feature Engineering ---
    combined_df['volume_change'] = combined_df.groupby('symbol')['volume'].pct_change(fill_method=None)
    combined_df['volume_zscore'] = (
        combined_df.groupby('symbol')['volume']
        .transform(lambda x: (x - x.rolling(14).mean()) / x.rolling(14).std())
    )
    combined_df['price_volume_corr'] = (
        combined_df.groupby('symbol')
        .apply(lambda g: g['close'].rolling(20).corr(g['volume']))
        .reset_index(level=0, drop=True)
    )

    # Replace inf/-inf with NaN to avoid scaler errors
    if np.isinf(combined_df.select_dtypes(include=[np.number])).any().any():
        logger.warning("Found inf or -inf in numeric columns, replacing with NaN.")
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 1. Fill RSI/MACD gaps per symbol after warm-up
    for symbol in combined_df['symbol'].unique():
        mask = combined_df['symbol'] == symbol
        idx = combined_df.loc[mask].index
        # Only fill after first valid value (warm-up)
        first_valid = combined_df.loc[idx, 'rsi'].first_valid_index()
        if first_valid is not None:
            combined_df.loc[idx, 'rsi'] = combined_df.loc[idx, 'rsi'].ffill().bfill()
            combined_df.loc[idx, 'macd'] = combined_df.loc[idx, 'macd'].ffill().bfill()

    # 2. Symbol-level truncation: keep only coins with enough valid rows
    MIN_ROWS = 50  # or your config
    valid_frames = []
    for symbol in combined_df['symbol'].unique():
        coin_df = combined_df[combined_df['symbol'] == symbol]
        coin_df = coin_df.dropna(subset=["rsi", "macd"])
        if len(coin_df) >= MIN_ROWS:
            valid_frames.append(coin_df)
        else:
            logger.warning(f"{symbol}: Only {len(coin_df)} valid rows after filling, skipping.")
    if not valid_frames:
        raise ValueError("No coins with enough valid rows after filling RSI/MACD.")
    combined_df = pd.concat(valid_frames, ignore_index=True)

    # 3. Preprocess: drop/fill missing features, generate/fill indicators as needed
    # Drop columns that are completely empty
    combined_df = combined_df.dropna(axis=1, how="all")
    # If any required feature is missing, fill with a default or drop those rows
    for col in REQUIRED_FEATURES:
        if col not in combined_df.columns:
            logger.warning(f"Missing column {col}, filling with NaN.")
            combined_df[col] = np.nan
    # Drop rows with missing RSI/MACD (warmup rows)
    before = len(combined_df)
    combined_df = combined_df.dropna(subset=["rsi", "macd"])
    logger.info(f"Dropped {before - len(combined_df)} rows with missing RSI/MACD.")
    # Drop rows with missing values in other required features except 'target'
    for col in REQUIRED_FEATURES:
        if col not in ["rsi", "macd", "target"]:
            before = len(combined_df)
            combined_df = combined_df[combined_df[col].notna()]
            logger.info(f"Dropped {before - len(combined_df)} rows with missing {col}.")
    # Drop rows with missing target
    if "target" not in combined_df.columns:
        logger.info("Generating 'target' column (e.g., next-period up/down label)...")
        # Example: Binary target if next close > current close
        combined_df = combined_df.sort_values(["symbol", "timestamp"])
        combined_df["target"] = (
            combined_df.groupby("symbol")["close"].shift(-1) > combined_df["close"]
        ).astype(float)
    combined_df = combined_df[combined_df["target"].notna()]
    if len(combined_df) == 0:
        raise ValueError("No rows left after dropping those with missing required features. Check your data quality!")

    # 4. Continue with your normal pipeline
    X, y, groups = preprocess_data(combined_df, drop_low_importance=True, importance_threshold=0.01)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns =X.columns)
    unique_groups = np.unique(groups)
    n_train = int(0.8 * len(unique_groups))
    train_groups = unique_groups[:n_train]
    test_groups = unique_groups[n_train:]
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Final Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}")
    save_model(model, scaler, MODEL_OUT_PATH)
    scaler_path = MODEL_OUT_PATH.replace(".pkl", "_scaler.pkl")
    save_scaler(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    custom_threshold = 0.52
    try:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_train_pred = (y_train_proba >= custom_threshold).astype(int)
        y_test_pred = (y_test_proba >= custom_threshold).astype(int)
    except:
        logger.warning("Model does not support predict_proba; falling back to default predict.")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = None
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        custom_threshold = 0.5
        y_test_pred = (model.predict_proba(X_test)[:,1] >= custom_threshold).astype(int)
    except:
        y_test_proba = None
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score, roc_curve
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(" Training Performance:")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Precision:", precision_score(y_train, y_train_pred))
    print("Recall:", recall_score(y_train, y_train_pred))
    print("F1 Score:", f1_score(y_train, y_train_pred))
    print("OOB Score:", model.oob_score_)
    print("\n Testing Performance:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall:", recall_score(y_test, y_test_pred))
    print("F1 Score:", f1_score(y_test, y_test_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_test_proba))
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(" Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    if y_test_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(" ROC Curve")
        plt.legend()
        plt.show()
    if hasattr(model, "feature_importances_"):
        col_names = df.drop(columns=["timestamp", "target", "symbol"]).columns
        feat_imp = pd.Series(model.feature_importances_, index=col_names).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.tight_layout()
        plt.show()
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting training script gracefully.")
        import sys
        sys.exit(0)

if __name__ == "__main__":
    try:
        refine_model_from_logs()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting trainer script gracefully.")
        import sys
        sys.exit(0)
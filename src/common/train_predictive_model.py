import pandas as pd
import os
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/mapping.json"))

def load_coin_data(symbol):
    csv_path = os.path.join(DATA_DIR, f"{symbol}_model_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No data for {symbol}")
    df = pd.read_csv(csv_path)
    return df

def load_historical_data(symbol, data_dir, start_date=None, end_date=None):
    csv_path = os.path.join(data_dir, f"{symbol}_model_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No data for {symbol}")
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]
    return df

def prepare_data(df):
    df = df.copy()
    # Target: will the next close be higher than the current close?
    df['target_up'] = (df['close'].shift(-1) > df['close']).astype(int)
    # Remove last row (no future close available)
    df = df.iloc[:-1]
    closes = df['close'].values  # Save closes for backtesting
    forbidden = {'target_up', 'timestamp', 'close'}
    features = df.select_dtypes(include='number').drop(forbidden, axis=1, errors='ignore')
    X = features
    y = df['target_up']
    return X, y, closes

def backtest_profit(y_true, y_pred, closes):
    balance = 100.0  # start with $100
    for i in range(len(y_pred)):
        if y_pred[i] == 1:  # "buy" signal
            if i + 1 < len(closes):
                balance *= closes[i + 1] / closes[i]
    return balance

def downsample(X, y):
    df = X.copy()
    df['target'] = y
    majority = df[df['target'] == 0]
    minority = df[df['target'] == 1]
    majority_downsampled = resample(majority, 
                                    replace=False, 
                                    n_samples=len(minority), 
                                    random_state=42)
    downsampled = pd.concat([majority_downsampled, minority])
    y_down = downsampled['target']
    X_down = downsampled.drop('target', axis=1)
    return X_down, y_down

def train_and_evaluate(X, y, closes, return_report=False):
    print("Class distribution:", y.value_counts(normalize=True))
    tscv = TimeSeriesSplit(n_splits=5)
    # For aggregate metrics
    accs, precs, recs, f1s = [], [], [], []
    reports = []  # To store classification reports for each fold
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        closes_test = closes[test_idx]
        # Downsample majority class
        X_train_ds, y_train_ds = downsample(X_train, y_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_ds, y_train_ds)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(report)  # Save report for this fold
        print(f"Fold {fold+1} classification report:")
        print(report)
        # Feature importance
        importances = pd.Series(model.feature_importances_, index=X.columns)
        print("Feature importances:")
        print(importances.sort_values(ascending=False))
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix Fold {fold+1}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        # Backtest
        profit = backtest_profit(y_test.values, y_pred, closes_test)
        print(f"Backtest final balance (starting from $100): ${profit:.2f}")
        # Aggregate metrics
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
    print("\nAggregate cross-validation performance:")
    print(f"Accuracy:  {sum(accs)/len(accs):.3f}")
    print(f"Precision: {sum(precs)/len(precs):.3f}")
    print(f"Recall:    {sum(recs)/len(recs):.3f}")
    print(f"F1 Score:  {sum(f1s)/len(f1s):.3f}")
    if return_report:
        return model, reports  # Return both the model and the detailed reports
    return model

def train_and_evaluate_all(X, y, closes, return_report=False):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42)
    }
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    for name, clf in classifiers.items():
        accs, precs, recs, f1s = [], [], [], []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            closes_test = closes[test_idx]
            # Downsample majority class
            X_train_ds, y_train_ds = downsample(X_train, y_train)
            # Skip fold if only one class present
            if len(np.unique(y_train)) < 2:
                print(f"Skipping fold for {name}: only one class in y_train.")
                continue
            clf.fit(X_train_ds, y_train_ds)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred, zero_division=0))
            recs.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
        results[name] = {
            "accuracy": sum(accs) / len(accs),
            "precision": sum(precs) / len(precs),
            "recall": sum(recs) / len(recs),
            "f1": sum(f1s) / len(f1s)
        }
    return results

def evaluate_all_coins():
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    for symbol in mapping.keys():
        try:
            df = load_coin_data(symbol)
            X, y, closes = prepare_data(df)
            train_and_evaluate(X, y, closes)
        except Exception as e:
            print(f"Skipping {symbol}: {e}")

def aggregate_all_coins_data(mapping_path=None):
    if mapping_path is None:
        mapping_path = MAPPING_PATH  # fallback to default if not provided
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    dfs = []
    for symbol in mapping.keys():
        try:
            df = load_coin_data(symbol)
            df['symbol'] = symbol  # Add symbol for reference
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {symbol}: {e}")
    if not dfs:
        raise ValueError("No coin data found!")
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

def test_on_single_coin(model, symbol):
    df = load_coin_data(symbol)
    X, y, closes = prepare_data(df)
    y_pred = model.predict(X)
    print(f"Test results for {symbol.upper()}:")
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {symbol.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def get_all_classifiers():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=10, min_samples_split=10, random_state=42
        ),
        "XGBoost": XGBClassifier(
            max_depth=3, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=10, min_samples_split=10, random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=3, min_child_samples=10, subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
        "LogisticRegression": LogisticRegression(C=0.1, max_iter=1000)
    }

def load_recent_coin_data(symbol, data_dir, hours=None):
    csv_path = os.path.join(data_dir, f"{symbol}_model_data.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns and hours is not None:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(hours=hours)
        df = df[df["timestamp"] >= cutoff]
    return df

# Usage example in main:
if __name__ == "__main__":
    # Aggregate all coins' data
    all_df = aggregate_all_coins_data()
    X, y, closes = prepare_data(all_df)

    # Filter data for the last 24 hours
    cutoff = datetime.now() - timedelta(hours=24)
    all_df = all_df[all_df["timestamp"] >= cutoff]

    # Optional: Scatter plot for debugging
    if "close" in all_df.columns and "ema_10" in all_df.columns:
        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(
            all_df["ema_10"], all_df["close"], 
            c=all_df["target_up"], cmap="coolwarm", alpha=0.7
        )
        plt.xlabel("EMA 10")
        plt.ylabel("Close Price")
        plt.title("All Coins: Close vs EMA 10 (colored by target_up)")
        plt.colorbar(scatter, label="target_up")
        plt.tight_layout()
        plt.show()

    model = train_and_evaluate(X, y, closes)

    # TimeSeriesSplit example
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # Skip fold if only one class present
        if len(np.unique(y_train)) < 2:
            print(f"Skipping fold: only one class in y_train.")
            continue
        if X_train.shape[1] == 0:
            print(f"Skipping fold: no usable features after filtering.")
            continue
        # Log class imbalance
        print(f"Train class distribution: {y_train.value_counts().to_dict()}")
        print(f"Test class distribution: {y_test.value_counts().to_dict()}")
        # Fit and evaluate your models here
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Log metrics as needed
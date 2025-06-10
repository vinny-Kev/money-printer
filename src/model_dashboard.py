import streamlit as st
import pandas as pd
import os
import json
import numpy as np
import glob
from train_predictive_model import get_all_classifiers, prepare_data, aggregate_all_coins_data, load_recent_coin_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/coin_mapping.json"))

st.title("Model Evaluation Dashboard (All Coins Aggregated)")

# --- Data Overview ---
df = aggregate_all_coins_data(MAPPING_PATH)
if df is None or df.empty:
    st.warning("No data found. Please check your data files.")
else:
    # Add coin feature
    if 'coin' not in df.columns:
        # If your aggregation doesn't already add it, do it here:
        # (Assumes each file is loaded with a coin name, adjust as needed)
        coin_dfs = []
        with open(MAPPING_PATH, "r") as f:
            mapping = json.load(f)
        for coin in mapping.keys():
            coin_df = load_recent_coin_data(coin, DATA_DIR, hours=None)
            if coin_df is not None and not coin_df.empty:
                coin_df['coin'] = coin
                coin_dfs.append(coin_df)
        df = pd.concat(coin_dfs, ignore_index=True)

    if 'close' in df.columns:
        df['close_pct_change'] = df.groupby('coin')['close'].pct_change()

    st.subheader("NaN Count Per Feature (after aggregation, before dropping)")
    st.dataframe(df.isna().sum())

    st.subheader("Aggregated DataFrame Shape")
    st.write(df.shape)

    X, y, closes = prepare_data(df)
    if 'RSI' in X.columns:
        X = X.drop(columns=['RSI']) 

    # Drop rows with NaNs in X or y
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]

    st.subheader("Feature Columns Used")
    st.write(X.columns.tolist())

    st.subheader("Class Distribution in y")
    st.info(f"{y.value_counts().to_dict()}")

    st.subheader("Total Rows Used")
    st.info(f"Total: {len(X)}")

    if len(X) < 10:
        st.warning("Not enough data for training/testing. Please collect more data.")
    else:
        # --- Model Evaluation ---
        tscv = TimeSeriesSplit(n_splits=5)
        classifiers = get_all_classifiers()
        test_metrics = []
        predictions = {}

        for name, clf in classifiers.items():
            train_accs, train_precs, train_recs, train_f1s = [], [], [], []
            test_accs, test_precs, test_recs, test_f1s = [], [], [], []
            y_tests, y_preds = [], []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
           
                if len(np.unique(y_train)) < 2:
                    st.warning(f"Skipping fold for {name}: only one class in y_train.")
                    continue
                clf.fit(X_train, y_train)
                # Train predictions
                y_train_pred = clf.predict(X_train)
                train_accs.append(accuracy_score(y_train, y_train_pred))
                train_precs.append(precision_score(y_train, y_train_pred, zero_division=0))
                train_recs.append(recall_score(y_train, y_train_pred, zero_division=0))
                train_f1s.append(f1_score(y_train, y_train_pred, zero_division=0))
                # Test predictions
                y_pred = clf.predict(X_test)
                test_accs.append(accuracy_score(y_test, y_pred))
                test_precs.append(precision_score(y_test, y_pred, zero_division=0))
                test_recs.append(recall_score(y_test, y_pred, zero_division=0))
                test_f1s.append(f1_score(y_test, y_pred, zero_division=0))
                y_tests.extend(y_test)
                y_preds.extend(y_pred)
            # Store average train and test metrics
            test_metrics.append({
                "Model": name,
                "Set": "Test",
                "Accuracy": np.mean(test_accs),
                "Precision": np.mean(test_precs),
                "Recall": np.mean(test_recs),
                "F1": np.mean(test_f1s)
            })
            test_metrics.append({
                "Model": name,
                "Set": "Train",
                "Accuracy": np.mean(train_accs),
                "Precision": np.mean(train_precs),
                "Recall": np.mean(train_recs),
                "F1": np.mean(train_f1s)
            })
            predictions[name] = y_preds

        st.subheader("Train & Test Set Performance (TimeSeriesSplit, avg over 5 splits)")
        metrics_df = pd.DataFrame(test_metrics).set_index(["Model", "Set"])
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"))

        st.subheader("Sample Predictions (last 20 rows, last split)")
        pred_df = pd.DataFrame({
            "actual": y_tests[-20:]
        })
        for name in classifiers.keys():
            pred_df[name] = predictions[name][-20:]
        st.dataframe(pred_df)

# --- Per-Coin Evaluation (Optional) ---
if st.checkbox("Evaluate Models for Each Coin Separately"):
    coins = []
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
        coins = list(mapping.keys())

    all_results = []
    for coin in coins:
        df = load_recent_coin_data(coin, DATA_DIR, hours=None)  # or your function to load all data for a coin
        if df is None or df.empty:
            continue
        X, y, closes = prepare_data(df)
        if 'RSI' in X.columns:
            X = X.drop(columns=['RSI'])
        mask = X.notnull().all(axis=1) & y.notnull()
        X = X[mask]
        y = y[mask]
        if len(X) < 8 or X.shape[1] == 0:
            st.warning(f"Skipping {coin}: not enough data or features after filtering.")
            continue
        tscv = TimeSeriesSplit(n_splits=5)
        classifiers = get_all_classifiers()
        for name, clf in classifiers.items():
            accs, precs, recs, f1s = [], [], [], []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accs.append(accuracy_score(y_test, y_pred))
                precs.append(precision_score(y_test, y_pred, zero_division=0))
                recs.append(recall_score(y_test, y_pred, zero_division=0))
                f1s.append(f1_score(y_test, y_pred, zero_division=0))
            all_results.append({
                "Coin": coin,
                "Model": name,
                "Accuracy": sum(accs)/len(accs),
                "Precision": sum(precs)/len(precs),
                "Recall": sum(recs)/len(recs),
                "F1": sum(f1s)/len(f1s)
            })

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        st.subheader("Per-Coin Model Performance (TimeSeriesSplit, avg over 5 splits)")
        st.dataframe(results_df)
    else:
        st.warning("No results to display. Not enough data per coin.")

# --- Stacking Meta-Model Training ---
st.subheader("Train Meta-Model on Stacked Features")
coins = []
with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)
    coins = list(mapping.keys())

meta_features = []
meta_targets = []
meta_coin = []

classifiers = get_all_classifiers()
model_names = list(classifiers.keys())

for coin in coins:
    df = load_recent_coin_data(coin, DATA_DIR, hours=None)
    if df is None or df.empty:
        continue
    if 'close' in df.columns:
        df['close_pct_change'] = df['close'].pct_change()
    X, y, closes = prepare_data(df)
    if 'RSI' in X.columns:
        X = X.drop(columns=['RSI'])
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]
    if len(X) < 8 or X.shape[1] == 0:
        st.warning(f"Skipping {coin}: not enough data or features after filtering.")
        continue
    tscv = TimeSeriesSplit(n_splits=5)
    # Prepare an array to hold oof predictions for all models for this coin
    oof_preds_matrix = np.zeros((len(X), len(model_names)))
    for m_idx, (name, clf) in enumerate(classifiers.items()):
        oof_preds = np.zeros(len(X))
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if len(np.unique(y_train)) < 2:
                st.warning(f"Skipping stacking fold for {coin} and {name}: only one class in y_train.")
                continue
            clf.fit(X_train, y_train)
            oof_preds[test_idx] = clf.predict(X_test)
        oof_preds_matrix[:, m_idx] = oof_preds
    meta_features.append(oof_preds_matrix)
    meta_targets.append(y.values)
    meta_coin.extend([coin] * len(y))

# Stack meta-features for meta-model
if meta_features:
    meta_X = np.vstack(meta_features)
    meta_y = np.concatenate(meta_targets)
    meta_coin = np.array(meta_coin)

    st.write("Meta-features and targets prepared for stacking model.")
    st.write(f"Meta-feature shape: {meta_X.shape}, Meta-target shape: {meta_y.shape}")

    # Optionally, add coin as a categorical feature
    meta_df = pd.DataFrame(meta_X, columns=[f"{name}_pred" for name in model_names])
    meta_df["coin"] = meta_coin

    # One-hot encode coin
    meta_df = pd.get_dummies(meta_df, columns=["coin"])

    # Train meta-model
    meta_model = LogisticRegression()
    meta_model.fit(meta_df, meta_y)
else:
    st.warning("No data available for training the meta-model. Ensure individual models have valid predictions.")

csv_files = glob.glob(os.path.join(DATA_DIR, "*_model_data.csv"))
coins = [os.path.basename(f).replace("_model_data.csv", "") for f in csv_files]

# After fitting a tree-based model, e.g. RandomForest
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    st.subheader(f"Feature Importance for {name}")
    st.dataframe(importance_df)
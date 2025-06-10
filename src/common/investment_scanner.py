import os
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
COIN_MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/coin_mapping.json"))
HOLD_LIST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/hold_list.json"))

def load_latest_data(symbol):
    csv_path = os.path.join(DATA_DIR, f"{symbol}_model_data.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df

def load_model(symbol):
    model_path = os.path.join(MODEL_DIR, f"{symbol}_rf_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

def save_hold_list(hold_list):
    with open(HOLD_LIST_PATH, "w") as f:
        json.dump(hold_list, f, indent=2)

def scan_coins():
    with open(COIN_MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    hold_list = []
    results = []
    for symbol in mapping.keys():
        df = load_latest_data(symbol)
        if df is None or len(df) < 20:
            continue
        # Prepare features as in training
        df['target_up'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        features = df.select_dtypes(include='number').drop(['target_up', 'timestamp'], axis=1, errors='ignore')
        X = features
        # Use last row for prediction
        latest_X = X.iloc[[-1]]
        # Load or train model
        model = load_model(symbol)
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            y = df['target_up']
            model.fit(X, y)
        pred = model.predict(latest_X)[0]
        action = "BUY" if pred == 1 else "SELL/SHORT"
        if action == "BUY":
            hold_list.append(symbol)
        results.append((symbol, action))
    save_hold_list(hold_list)
    return results

if __name__ == "__main__":
    results = scan_coins()
    print("Investment Decisions:")
    print("{:<10} {:<10}".format("SYMBOL", "ACTION"))
    print("-" * 22)
    for symbol, action in results:
        print("{:<10} {:<10}".format(symbol.upper(), action))
    # Print hold list summary
    if os.path.exists(HOLD_LIST_PATH):
        with open(HOLD_LIST_PATH, "r") as f:
            hold_list = json.load(f)
        print("\nCurrent HOLD list (will not be deleted):")
        print(", ".join([s.upper() for s in hold_list]))
    else:
        print("\nNo coins currently on HOLD.")
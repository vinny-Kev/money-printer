import os
import json
from train_predictive_model import load_coin_data, prepare_data, train_and_evaluate
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/coin_mapping.json"))
LIVE_STATS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/live_stats.json"))


    
def get_trade_fraction(balance):
    if balance < 100:
        return 0.25
    elif balance < 1000:
        return 0.15
    elif balance < 10000:
        return 0.10
    else:
        return 0.05


def simulate_live_trading():
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    total_trades = 0
    wins = 0
    balance = 100.0  # Start with $100

    for symbol in mapping.keys():
        try:
            df = load_coin_data(symbol)
            X, y, closes = prepare_data(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            for i in range(len(y_test)):
                total_trades += 1
                if y_pred[i] == y_test.values[i]:
                    wins += 1
                # Simulate gain/loss (simple version: only act on "buy" signals)
                if y_pred[i] == 1 and i+1 < len(y_test):
                    trade_fraction = get_trade_fraction(balance)
                    balance += balance * trade_fraction * (closes[-len(y_test)+i+1] / closes[-len(y_test)+i] - 1)
        except Exception as e:
            continue

    winrate = wins / total_trades if total_trades > 0 else 0.0
    gain = balance - 100.0

    stats = {"winrate": winrate, "gain": gain}
    with open(LIVE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Live trading simulation complete. Winrate: {winrate:.2%}, Gain: ${gain:.2f}")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    simulate_live_trading()
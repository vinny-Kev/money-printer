import os
import json
import time
from common.train_predictive_model import load_coin_data, prepare_data, load_model
from sklearn.ensemble import RandomForestClassifier
from common.investment_caller import simulate_paper_trading

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/coin_mapping.json"))
PAPER_TRADES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trades/paper_trades.json"))
TRADES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trades"))
TRADE_HISTORY_PATH = os.path.join(TRADES_DIR, "paper_trades.json")
PAPER_STATS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/paper_stats.json"))

def get_trade_fraction(balance):
    if balance < 100:
        return 0.25
    elif balance < 1000:
        return 0.15
    elif balance < 10000:
        return 0.10
    else:
        return 0.05

def simulate_paper_trading():
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    total_trades = 0
    wins = 0
    balance = 10.0  # Start with $10
    trade_history = []

    os.makedirs(TRADES_DIR, exist_ok=True)

    # Load previous trade history if you want to continue (optional)
    if os.path.exists(TRADE_HISTORY_PATH):
        with open(TRADE_HISTORY_PATH, "r") as f:
            try:
                trade_history = json.load(f)
            except Exception:
                trade_history = []

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
                correct = y_pred[i] == y_test.values[i]
                if correct:
                    wins += 1
                if y_pred[i] == 1 and i+1 < len(y_test):
                    trade_fraction = get_trade_fraction(balance)
                    trade_amount = balance * trade_fraction
                    price_buy = closes[-len(y_test)+i]
                    price_sell = closes[-len(y_test)+i+1]
                    price_change = price_sell / price_buy
                    profit = trade_amount * (price_change - 1)
                    balance += profit
                    trade_history.append({
                        "timestamp": time.time(),
                        "symbol": symbol,
                        "index": int(i),
                        "action": "BUY",
                        "trade_amount": float(trade_amount),
                        "price_buy": float(price_buy),
                        "price_sell": float(price_sell),
                        "profit": float(profit),
                        "balance_after": float(balance),
                        "model_prediction": int(y_pred[i]),
                        "actual_label": int(y_test.values[i]),
                        "correct": bool(correct)
                    })
        except Exception as e:
            continue

    winrate = wins / total_trades if total_trades > 0 else 0.0
    gain = balance - 10.0

    stats = {"winrate": winrate, "gain": gain, "final_balance": balance}
    with open(PAPER_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    with open(TRADE_HISTORY_PATH, "w") as f:
        json.dump(trade_history, f, indent=2)
    print(f"[PaperTrader] Simulation complete. Winrate: {winrate:.2%}, Gain: ${gain:.2f}, Final Balance: ${balance:.2f}")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    simulate_paper_trading()

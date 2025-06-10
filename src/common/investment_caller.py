import os
import json
import pandas as pd
import time
from common.train_predictive_model import load_coin_data, prepare_data, load_model
from sklearn.ensemble import RandomForestClassifier
from common.investment_caller import simulate_paper_trading

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/coin_mapping.json"))
TRADES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trades"))
PAPER_TRADES_PATH = os.path.join(TRADES_DIR, "paper_trades.csv")

def get_trade_params(price, action, tp_pct=0.05, sl_pct=0.02):
    if action == "BUY":
        take_profit = price * (1 + tp_pct)
        stop_loss = price * (1 - sl_pct)
    elif action == "SELL/SHORT":
        take_profit = price * (1 - tp_pct)
        stop_loss = price * (1 + sl_pct)
    else:
        take_profit = stop_loss = None
    return take_profit, stop_loss

def simulate_paper_trading():
    os.makedirs(TRADES_DIR, exist_ok=True)
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    trades = []
    for symbol in mapping.keys():
        try:
            df = load_coin_data(symbol)
            X, y, closes = prepare_data(df)
            if len(X) == 0:
                continue
            # Use the last row for prediction
            latest_X = X.iloc[[-1]]
            latest_price = closes[-1]
            # Use your trained model or train a new one
            model = load_model(symbol)
            if model is None:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
            pred = model.predict(latest_X)[0]
            action = "BUY" if pred == 1 else "SELL/SHORT"
            if action == "HOLD":
                continue
            take_profit, stop_loss = get_trade_params(latest_price, action)
            trade = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "action": action,
                "price": latest_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss
            }
            trades.append(trade)
            print(f"[PaperTrader] {action} {symbol.upper()} at {latest_price:.4f} TP: {take_profit:.4f} SL: {stop_loss:.4f}")
        except Exception as e:
            print(f"[PaperTrader] Skipping {symbol}: {e}")
    # Save trades to CSV
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(PAPER_TRADES_PATH, index=False)
    print(f"[PaperTrader] Logged {len(trades)} trades to {PAPER_TRADES_PATH}")

if __name__ == "__main__":
    simulate_paper_trading()
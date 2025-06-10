import json
from transaction_manager import buy_coin, sell_coin  # You need to implement these!

CALLS_PATH = "investment_calls.json"

def handle_investment_calls():
    with open(CALLS_PATH, "r") as f:
        calls = json.load(f)
    for call in calls:
        symbol = call["symbol"].lower()
        action = call["action"]
        print(f"Handling {action} for {symbol}")
        if action == "BUY":
            buy_coin(symbol)
        elif action == "SELL/SHORT":
            sell_coin(symbol)
        else:
            print(f"No action for {symbol}")

if __name__ == "__main__":
    handle_investment_calls()
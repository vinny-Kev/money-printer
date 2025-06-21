import os
import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

MODEL_PATH = os.path.join(BASE_DIR, "data/models/trained_model.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data/scraped_data")
# ADJUST PARAMS AS NEEDED
# These parameters can be adjusted based on your trading strategy and risk tolerance
INITIAL_CAPITAL = 17.45
PARTIAL_EXIT_THRESHOLD = 100
CONFIDENCE_THRESHOLD = 0.5  # Only trade if model confidence > 50% change to 60 for real trading
PROFIT_THRESHOLD = 0.05  # 5%
STOP_LOSS = -0.025  # -2.5%
PARTIAL_TAKE_PROFIT = 0.5  # take half profits at 5%
FEATURE_PATH = os.path.join(BASE_DIR, "data/models/important_features.json")
SLIPPAGE_RATE = 0.002  # 0.2%
PLATFORM_FEE = 0.001   # 0.1%
GAS_FEE = 0.15         # fixed fee in USD
MIN_EXPECTED_GAIN = 0.05  # Only trade if expected price movement > 5%


def get_required_gain(trade_amount):
    """Returns minimum % gain required to cover slippage, platform fee, and gas"""
    total_fees = (PLATFORM_FEE * 2 * trade_amount) + GAS_FEE
    slippage_loss = SLIPPAGE_RATE * 2  # 2x for entry/exit

    return (total_fees / trade_amount) + slippage_loss

with open(FEATURE_PATH, "r") as f:
    important_features = json.load(f)
def get_trade_fraction(balance):
    if balance < 100:
        return 0.25
    elif balance < 1000:
        return 0.15
    elif balance < 10000:
        return 0.10
    else:
        return 0.05
    
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def simulate_trade(row, prediction, current_price, trade_amount, capital, confidence=None):
    future_price = row["close"]
    change = (future_price - current_price) / current_price
    slippage_loss = SLIPPAGE_RATE * 2
    adjusted_change = change - slippage_loss

    fee_total = PLATFORM_FEE * 2 * trade_amount
    gas_fee = GAS_FEE

    use_partial_exit = capital >= PARTIAL_EXIT_THRESHOLD

    if prediction == 1:  # LONG
        if adjusted_change >= PROFIT_THRESHOLD:
            if use_partial_exit:
                profit = trade_amount * (PROFIT_THRESHOLD * PARTIAL_TAKE_PROFIT)
                remaining = trade_amount * (1 - PARTIAL_TAKE_PROFIT)
                extra_gain = max(min(adjusted_change, 0.3) - PROFIT_THRESHOLD, 0)
                profit += remaining * extra_gain
            else:
                profit = trade_amount * adjusted_change
        elif adjusted_change <= STOP_LOSS:
            profit = trade_amount * STOP_LOSS
        else:
            profit = trade_amount * adjusted_change
    else:  # SHORT
        if adjusted_change <= -PROFIT_THRESHOLD:
            if use_partial_exit:
                profit = trade_amount * (PROFIT_THRESHOLD * PARTIAL_TAKE_PROFIT)
                remaining = trade_amount * (1 - PARTIAL_TAKE_PROFIT)
                extra_gain = max(min(-adjusted_change, 0.3) - PROFIT_THRESHOLD, 0)
                profit += remaining * extra_gain
            else:
                profit = -trade_amount * adjusted_change
        elif adjusted_change >= abs(STOP_LOSS):
            profit = trade_amount * (-STOP_LOSS)
        else:
            profit = -trade_amount * adjusted_change

    profit -= fee_total
    profit -= gas_fee

    return profit, fee_total, gas_fee, slippage_loss * trade_amount





def main():
    model = load_model()
    capital = INITIAL_CAPITAL
    results = []

    for file_path in glob.glob(os.path.join(DATA_DIR, "*_model_data.csv")):
        df = pd.read_csv(file_path)
        symbol = os.path.basename(file_path).split("_")[0]

        df = df.dropna().reset_index(drop=True)
        if df.shape[0] < 31:
            continue

        test_data = df.head(31).copy()
        current_row = test_data.iloc[0]
        future_row = test_data.iloc[-1]
        current_price = current_row["close"]
        next_price = future_row["close"]
        trade_amount = capital * get_trade_fraction(capital)
        expected_gain = (next_price - current_price) / current_price
        required_gain = get_required_gain(trade_amount)

        if abs(expected_gain) < required_gain:
            print(f"Skipping {symbol} — Expected gain {expected_gain:.2%} < required {required_gain:.2%} to break even")
            continue

        # --- Feature processing ---
        test_data["symbol_id"] = symbol

        for col in important_features:
            if col not in test_data.columns:
                test_data[col] = 0  # pad missing with 0

        features = test_data[important_features].copy()
        features = features.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # --- Prediction & simulation ---
        probas = model.predict_proba([features[0]])[0]
        confidence = probas[1] if probas[1] > 0.5 else 1 - probas[1]
        prediction = np.argmax(probas)

        # Skip if not confident enough
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"⏭ Skipping {symbol} — Low confidence: {confidence:.2f}")
            continue
   
        trade_amount = capital * get_trade_fraction(capital)
        profit, fee_total, gas_fee, slippage_cost = simulate_trade(
            future_row, prediction, current_price, trade_amount, capital, confidence
        )
        capital += profit

        # --- Log result ---
        results.append({
            "symbol": symbol,
            "pred": prediction,
            "start_price": current_price,
            "end_price": next_price,
            "expected_gain": expected_gain,
            "confidence": confidence,
            "trade_amount": trade_amount,
            "platform_fee": fee_total,
            "gas_fee": gas_fee,
            "slippage_loss": slippage_cost,
            "gross_profit_before_fees": profit + fee_total + gas_fee,
            "profit": profit,
            "capital": capital
        })

    # --- Final summary ---
    df_result = pd.DataFrame(results)
    df_result.to_csv("paper_trader_results.csv", index=False)
    print(df_result)
    print(f"Final capital: ${capital:.2f} | PnL: ${capital - INITIAL_CAPITAL:.2f}")


if __name__ == "__main__":
    main()

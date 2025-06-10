import os
import json
import time
import pandas as pd
import requests_cache
from data_scraper import get_markets_batch, needs_update
from sklearn.model_selection import GroupKFold
import glob
import subprocess

# Cache all requests for 1 hour
requests_cache.install_cache('coingecko_cache', expire_after=3600)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/coin_mapping.json"))

BATCH_SIZE = 50  # CoinGecko allows up to 250, but 50 is safe for most endpoints

ESSENTIAL_COINS = ["btc", "eth", "usdt", "sol"]  # Add more coins to keep if needed

def load_mapping():
    with open(MAPPING_PATH, "r") as f:
        return json.load(f)

def save_mapping(mapping):
    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)

def fetch_top_100_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch top coins: {response.text}")
    data = response.json()
    return {coin["symbol"].lower(): coin["id"] for coin in data}

def update_coin_mapping():
    top_coins = fetch_top_100_coins()
    mapping = {sym: cid for sym, cid in top_coins.items()}
    # Always keep essentials
    for sym in ESSENTIAL_COINS:
        if sym not in mapping:
            mapping[sym] = sym  # fallback to symbol as id if not present
    save_mapping(mapping)
    print(f"Updated coin mapping with {len(mapping)} coins.")

def batch_fetch_and_save_market_data(symbols, out_path):
    print(f"[Batch] Fetching market data for {len(symbols)} coins...")
    all_batches = []
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        try:
            batch_df = get_markets_batch(batch, vs_currency="usd")
            all_batches.append(batch_df)
            print(f"[Batch] Got {len(batch_df)} coins in batch {i//BATCH_SIZE+1}")
        except Exception as e:
            print(f"[Batch] Failed batch {batch}: {e}")
        time.sleep(2)  # avoid rate limits
    if all_batches:
        full_df = pd.concat(all_batches, ignore_index=True)
        full_df.to_csv(out_path, index=False)
        print(f"[Batch] Saved market data to {out_path}")
    else:
        print("[Batch] No market data fetched.")

def main():
    mapping = load_mapping()
    symbols = list(mapping.keys())
    market_data_path = os.path.join(DATA_DIR, "all_coins_market_data.csv")
    batch_fetch_and_save_market_data(symbols, market_data_path)

    # Now, for coins that need full OHLC/history, call per-coin scraper
    for symbol in symbols:
        output_path = os.path.join(DATA_DIR, f"{symbol}_model_data.csv")
        if not needs_update(output_path):
            print(f"[Scraper] Skipping {symbol}: already up-to-date.")
            continue
        
        print(f"[Scraper] Scraping full OHLC/history for {symbol}...")
        
        result = subprocess.run(
            ["python", "data_scraper.py", symbol, "90"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"[Scraper] Error scraping {symbol}: {result.stderr}")
        else:
            print(f"[Scraper] Successfully scraped {symbol}")
        time.sleep(5)  # avoid rate limits



    all_files = glob.glob(os.path.join(DATA_DIR, "*_model_data.csv"))
    dfs = [pd.read_csv(f) for f in all_files]
    full_df = pd.concat(dfs, ignore_index=True)

    feature_columns = [col for col in full_df.columns if col not in ["target", "coin", "timestamp"]]

    X = full_df[feature_columns]  
    y = full_df["target"]         
    groups = full_df["coin"]      

    gkf = GroupKFold(n_splits=5)
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # Now, all data for a coin is only in train or test, never both!

if __name__ == "__main__":
    main()
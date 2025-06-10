import requests
import json
import os
import time

BASE_URL = "https://api.coingecko.com/api/v3"
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "coin_mapping.json"))
HOLD_LIST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "hold_list.json"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
ESSENTIAL_COINS = {"btc", "eth", "sol"}  # always keep these

HEADERS = {
    "accept": "application/json"
}

def load_existing_mapping():
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r") as f:
            return json.load(f)
    return {}

def fetch_top_100_coins():
    url = f"{BASE_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch top coins: {response.text}")
    data = response.json()
    return {coin["symbol"].lower(): coin["id"] for coin in data}

def update_mapping():
    existing_map = load_existing_mapping()
    top_coins = fetch_top_100_coins()

    # Start with always-keep symbols
    updated_map = {sym: existing_map[sym] for sym in existing_map if sym in ESSENTIAL_COINS}

    # Add or update top 100 coins
    updated_map.update(top_coins)

    with open(MAPPING_PATH, "w") as f:
        json.dump(updated_map, f, indent=2)
    print(f"[✓] Coin mapping updated. Total entries: {len(updated_map)}")

def clean_unused_coins():
    # Load hold list and mapping
    with open(HOLD_LIST_PATH, "r") as f:
        hold_list = set(json.load(f))
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    # Determine coins to keep
    coins_to_keep = set(hold_list) | ESSENTIAL_COINS

    # Remove unused coins from mapping
    new_mapping = {k: v for k, v in mapping.items() if k in coins_to_keep}
    with open(MAPPING_PATH, "w") as f:
        json.dump(new_mapping, f, indent=2)

    # Remove unused model files
    for fname in os.listdir(DATA_DIR):
        if fname.endswith("_model_data.csv"):
            symbol = fname.replace("_model_data.csv", "")
            if symbol not in coins_to_keep:
                os.remove(os.path.join(DATA_DIR, fname))
                print(f"Removed unused model file: {fname}")

    print(f"Cleaned mapping and models. Kept: {sorted(coins_to_keep)}")

def remove_old_model_files(data_dir, hours=24):
    now = time.time()
    cutoff = now - hours * 3600
    for fname in os.listdir(data_dir):
        if fname.endswith("_model_data.csv"):
            fpath = os.path.join(data_dir, fname)
            if os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                print(f"Removed old model file: {fname}")

# Call this in your main or management routine
remove_old_model_files(DATA_DIR, hours=24)

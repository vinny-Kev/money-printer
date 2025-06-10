# from symtable import Symbol
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json
import requests_cache

 ## TODO: Add error handling for network issues, API rate limits, etc.
 # Instead of redundantly using time.sleep() after each request,
 # we can use a single sleep at the end of each batch processing by measuring actual request frequency and applying exponential backoff only when we get 
 # rate-limited by the API.(ERROR 429 Too Many Requests)
 # Issue: overfitting solution: we can implement a rolling window approach to only keep the most recent data points for each coin. 
 # Solution 2:we should also use 5 minute intervals of data instead of 30 mins interval for more to increase the granularity of the data hopefully reducing overfitting.
 # issue: inefficient merging of OHLC and market chart data, especially for large datasets. solution: use pd.merge_asof() to merge on timestamp with nearest match, which is more efficient for time series data. also implement a file cache system where merged + enriched data is saved to disk after processing each coin, so we don't have to recompute it every time is only recomputed if raw components are new
 
 # This script scrapes cryptocurrency data from the CoinGecko API, processes it, and saves it to CSV files.

# Load API key from .env or manually set here
API_KEY = os.getenv("COINGECKO_API_KEY") or "CG-15KeEDfysNvemfAy69ySby8s"

BASE_URL = "https://api.coingecko.com/api/v3"

HEADERS = {
    "accept": "application/json"
    # "x-cg-pro-api-key": API_KEY
}

requests_cache.install_cache('coingecko_cache', expire_after=3600)  # cache for 1 hour

def load_coin_mapping(path="../configs/coin_mapping.json"):
    if not os.path.exists(path):
        print(f"Warning: Mapping file {path} not found. Using empty mapping.")
        return {}
    with open(path, "r") as f:
        return json.load(f)

def get_coin_id(symbol, mapping):
    symbol_lower = symbol.lower()
    if symbol_lower in mapping:
        return mapping[symbol_lower]
    
    # Fallback: full list search (slower)
    url = f"{BASE_URL}/coins/list"
    response = requests.get(url, headers=HEADERS)
    coins_list = response.json()
    
    for coin in coins_list:
        if coin["symbol"].lower() == symbol_lower:
            return coin["id"]
    
    raise Exception(f"Coin symbol '{symbol}' not found")

def get_ohlc(coin_id, vs_currency="usd", days=1):
    url = f"{BASE_URL}/coins/{coin_id}/ohlc?vs_currency={vs_currency}&days={days}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Error fetching OHLC data: {response.text}")
    
    raw_data = response.json()
    df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def get_market_chart(coin_id, vs_currency="usd", days=1):
    params = {
        "vs_currency": "usd",
        "days": days
        # Do NOT include "interval"
    }
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching market chart: {response.text}")
    data = response.json()

    # Extract prices, market_caps, total_volumes
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    market_caps = pd.DataFrame(data.get("market_caps", []), columns=["timestamp", "market_cap"])
    total_volumes = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp", "total_volume"])

    # Merge all on timestamp
    df = prices
    if not market_caps.empty:
        df = df.merge(market_caps, on="timestamp", how="left")
    if not total_volumes.empty:
        df = df.merge(total_volumes, on="timestamp", how="left")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def get_markets_batch(symbols, vs_currency="usd"):
    ids = ",".join(symbols)
    url = f"{BASE_URL}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "ids": ids,
        "order": "market_cap_desc",
        "per_page": len(symbols),
        "page": 1,
        "sparkline": False
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching batch market data: {response.text}")
    return pd.DataFrame(response.json())

def get_coin_indicators(df):
    df = df.copy()
    df['EMA_10'] = df['price'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['price'].ewm(span=30, adjust=False).mean()
    df['RSI'] = compute_rsi(df['price'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_fvg_and_engulfing(df):
    df = df.copy()
    # Fair Value Gap
    df['fvg_up'] = df['low'] > df['high'].shift(1)
    df['fvg_down'] = df['high'] < df['low'].shift(1)
    df['fair_value_gap'] = df['fvg_up'] | df['fvg_down']

    # Engulfing Candles
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    # Bullish Engulfing
    df['bullish_engulfing'] = (
        (df['close'] > df['open']) &
        (prev_close < prev_open) &
        (df['close'] > prev_open) &
        (df['open'] < prev_close)
    )
    # Bearish Engulfing
    df['bearish_engulfing'] = (
        (df['close'] < df['open']) &
        (prev_close > prev_open) &
        (df['open'] > prev_close) &
        (df['close'] < prev_open)
    )
    return df

def get_coin_details(coin_id):
    url = f"{BASE_URL}/coins/{coin_id}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Error fetching coin details: {response.text}")
    data = response.json()
    return {
        "circulating_supply": data.get("market_data", {}).get("circulating_supply"),
        "total_supply": data.get("market_data", {}).get("total_supply"),
        "ath": data.get("market_data", {}).get("ath", {}).get("usd"),
        "ath_change_percentage": data.get("market_data", {}).get("ath_change_percentage", {}).get("usd"),
        "atl": data.get("market_data", {}).get("atl", {}).get("usd"),
        "atl_change_percentage": data.get("market_data", {}).get("atl_change_percentage", {}).get("usd"),
        "sentiment_votes_up_percentage": data.get("sentiment_votes_up_percentage"),
        "sentiment_votes_down_percentage": data.get("sentiment_votes_down_percentage"),
        "twitter_followers": data.get("community_data", {}).get("twitter_followers"),
        "reddit_subscribers": data.get("community_data", {}).get("reddit_subscribers"),
        "github_stars": data.get("developer_data", {}).get("stars"),
        "github_forks": data.get("developer_data", {}).get("forks"),
        "github_subscribers": data.get("developer_data", {}).get("subscribers"),
        "github_total_issues": data.get("developer_data", {}).get("total_issues"),
        "github_closed_issues": data.get("developer_data", {}).get("closed_issues"),
        "github_pull_requests_merged": data.get("developer_data", {}).get("pull_requests_merged"),
        "github_pull_request_contributors": data.get("developer_data", {}).get("pull_request_contributors"),
    }

def main():
    coin_mapping = load_coin_mapping()
    symbols = list(coin_mapping.keys())
    batch_size = 50  # CoinGecko allows up to 250, but 50 is safe for most endpoints

    # Batch fetch market data for all coins
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            market_df = get_markets_batch(batch, vs_currency="usd")
            print(market_df[["id", "symbol", "market_cap", "total_volume"]].head())
            # You can save or merge this info with your per-coin data as needed
        except Exception as e:
            print(f"[Scraper] Batch fetch failed for {batch}: {e}")

        # --- FIX: process each symbol in this batch here ---
        for symbol in batch:
            try:
                coin_id = get_coin_id(symbol, coin_mapping)
                print(f"[Scraper] Processing {symbol} (id: {coin_id})")
                ohlc_df = get_ohlc(coin_id, days=2)
                time.sleep(1)
                chart_df = get_market_chart(coin_id, days=2)
                time.sleep(1)
                merged_df = pd.merge_asof(
                    ohlc_df.sort_values('timestamp'),
                    chart_df.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest'
                )
                enriched_df = get_coin_indicators(merged_df)
                enriched_df = add_fvg_and_engulfing(enriched_df)
                output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{symbol}_model_data.csv")
                enriched_df.to_csv(output_path, index=False)
                print(f"[Scraper] Saved enriched data for {symbol} to {output_path}")
            except Exception as e:
                print(f"[Scraper] Failed to process {symbol}: {e}")

def needs_update(output_path, max_age_hours=24, min_hours_coverage=24):
    if not os.path.exists(output_path):
        return True
    try:
        df = pd.read_csv(output_path, parse_dates=["timestamp"])
        if df.empty or "timestamp" not in df.columns:
            return True
        # Check time span covered by the data
        min_time = df["timestamp"].min()
        max_time = df["timestamp"].max()
        hours_covered = (pd.to_datetime(max_time) - pd.to_datetime(min_time)).total_seconds() / 3600
        if hours_covered < min_hours_coverage:
            print(f"[Scraper] {output_path} only covers {hours_covered:.1f}h, needs update.")
            return True
    except Exception as e:
        print(f"[Scraper] Error reading {output_path}: {e}")
        return True
    # Also check file age as a fallback
    file_age_hours = (time.time() - os.path.getmtime(output_path)) / 3600
    return file_age_hours >= max_age_hours

def continuous_scrape_loop(interval=300):
    print(f"[Scraper] Starting continuous scraping loop every {interval} seconds...")
    import importlib
    while True:
        try:
            main()  # Scrape all coins needing update
            # --- Trigger trainer after scraping ---
            try:
                trainer = importlib.import_module("trainer")
                if hasattr(trainer, "train_once"):
                    print("[Scraper] Triggering immediate training after scrape loop...")
                    trainer.train_once()
                else:
                    print("[Scraper] No train_once() found in trainer.")
            except Exception as e:
                print(f"[Scraper] Failed to trigger trainer: {e}")
        except Exception as e:
            print(f"[Scraper] Error in scraping loop: {e}")
        print(f"[Scraper] Sleeping for {interval} seconds...\n")
        time.sleep(interval)

if __name__ == "__main__":
    import sys
    coin_mapping = load_coin_mapping()
    if len(sys.argv) > 2:
        coin_symbol = sys.argv[1]
        days = int(sys.argv[2])
    elif len(sys.argv) > 1:
        coin_symbol = sys.argv[1]
        days = 365
    else:
        coin_symbol = "sol"
        days = 365

    coin_id = get_coin_id(coin_symbol, coin_mapping)
    print(f"Coin ID for {coin_symbol}: {coin_id}")

    try:
        ohlc_df = get_ohlc(coin_id, days=days)
        time.sleep(3)  
    except Exception as e:
        print(f"Failed to scrape {coin_symbol}: {e}")
        sys.exit(1)
    chart_df = get_market_chart(coin_id, days=days)
    time.sleep(3) 
    print("Raw Chart Data:")
    print(chart_df.tail())

    # Merge OHLC with market chart data on timestamp
    merged_df = pd.merge_asof(
        ohlc_df.sort_values('timestamp'),
        chart_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )

    enriched_df = get_coin_indicators(merged_df)
    enriched_df = add_fvg_and_engulfing(enriched_df)
    print("With Indicators, FVG, and Engulfing:")
    print(enriched_df.tail())

    # Save to ../data/models/
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))



    output_path = os.path.join(output_dir, f"{coin_symbol}_model_data.csv")
    os.makedirs(output_dir, exist_ok=True)
    enriched_df.to_csv(output_path, index=False)
    print(f"Saved enriched data to {output_path}")

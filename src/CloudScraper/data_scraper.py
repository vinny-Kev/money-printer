import os
import logging
import backoff
import requests
import pandas as pd
from binance import ThreadedWebsocketManager, Client
from bucket_uploader import (
    upload_file,
    check_and_pause_if_bucket_full,
    check_bucket_storage,
    check_daily_upload_limit,
    check_total_upload_limit,
    data_bucket
)
from datetime import datetime, timezone
from io import BytesIO
import signal
import sys
import time
import random
import json

# Create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Logger setup
log_filename = os.path.join(LOG_DIR, f"log_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    logger.error("Missing Binance API keys in environment variables.")
    sys.exit(1)

# Initialize Binance client and WebSocket manager
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

# Global running flag for graceful shutdown
running = True

def handle_sigint(signal_received, frame):
    """
    Handle SIGINT and SIGTERM signals for graceful shutdown.
    """
    global running
    logger.info("🛑 Gracefully stopping scraper...")
    send_discord_alert("🛑 **Scraper Stopped**: The scraper is shutting down.")
    running = False

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# Discord alert function
def send_discord_alert(message):
    """
    Send an alert message to a Discord channel via webhook.
    """
    scraper_quotes = [
        "Yes My Lord! I'm on it!",
        "I will now gather information, my lord.",
        "The data shall be collected, my lord.",
        "The bytes are ready, my lord.",
        "The data is being gathered, my lord.",
    ]
    quote = random.choice(scraper_quotes)
    if not DISCORD_WEBHOOK:
        logger.warning("⚠️ No Discord webhook set. Skipping alert.")
        return

    try:
        payload = {
            "content": f"🏯 **{quote}**\n\n{message}"
        }
        response = requests.post(DISCORD_WEBHOOK, json=payload)
        if response.status_code == 204:
            logger.info("✅ Discord alert sent.")
        else:
            logger.warning(f"❌ Failed to send Discord alert: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Exception while sending Discord alert: {e}")

# Save data to a single Parquet file per coin
def save_data_to_parquet(df, symbol):
    filename = f"data/{symbol}.parquet"
    try:
        if os.path.exists(filename):
            existing_df = pd.read_parquet(filename)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp")
        else:
            combined_df = df

        # ✅ Fix: Ensure timestamp is in correct format
        combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])

        combined_df.to_parquet(filename, index=False, compression="snappy")
        logger.info(f"✅ Data for {symbol} saved to {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save data for {symbol} to {filename}: {e}")


def save_all_to_parquet():
    """
    Save all OHLCV data to Parquet files and upload to the data bucket.
    """
    try:
        for symbol, data in ohlcv_buffer.items():
            if not data:
                continue

            # Convert to DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                logger.warning(f"[WARN] Skipped saving: OHLCV data for {symbol} is empty.")
                continue

            # Save data to a single Parquet file per coin
            save_data_to_parquet(df, symbol)

            # Clear the buffer for the symbol
            ohlcv_buffer[symbol].clear()
    except Exception as e:
        logger.error(f"❌ Failed to save OHLCV data to Parquet: {e}")

# Cache directory and file
CACHE_DIR = "ohlcv_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "ohlcv_buffer.json")

def save_ohlcv_buffer_to_disk():
    """
    Save the OHLCV buffer to disk as a JSON file.
    """
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(ohlcv_buffer, f, default=str)  # Convert non-serializable objects to strings
        logger.info(f"✅ OHLCV buffer saved to disk at {CACHE_FILE}")
    except Exception as e:
        logger.error(f"❌ Failed to save OHLCV buffer to disk: {e}")

def load_ohlcv_buffer_from_disk():
    """
    Load the OHLCV buffer from disk if it exists.
    """
    global ohlcv_buffer
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                ohlcv_buffer = json.load(f)
            logger.info(f"✅ OHLCV buffer loaded from disk at {CACHE_FILE}")
        except Exception as e:
            logger.error(f"❌ Failed to load OHLCV buffer from disk: {e}")
            ohlcv_buffer = {}
    else:
        logger.info("No existing OHLCV buffer found on disk. Starting fresh.")
        ohlcv_buffer = {}

# Graceful shutdown handling
def signal_handler(sig, frame):
    logger.info("Shutting down scraper...")
    send_discord_alert("🛑 **Scraper Stopped**: The scraper has been shut down.")
    save_all_to_parquet()  # Save OHLCV data before exiting
    save_ohlcv_buffer_to_disk()  # Save buffer to disk before exiting
    twm.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Sleep until the next interval (e.g., 0:00, 5:00, 10:00).
def sleep_until_next_interval(interval_sec=300):
    """
    Sleep until the next interval (e.g., 0:00, 5:00, 10:00).
    :param interval_sec: Interval in seconds (default: 300 seconds = 5 minutes).
    """
    now = time.time()
    sleep_time = interval_sec - (now % interval_sec)
    logger.info(f"Sleeping for {sleep_time:.2f} seconds until the next interval...")
    time.sleep(sleep_time)

# Check if the bucket is full and pause the scraper if necessary.
def check_and_pause_if_bucket_full():
    """
    Check if the bucket is full and pause the scraper if necessary.
    """
    try:
        storage_used, storage_limit = check_bucket_storage(bucket_type="data")
        if storage_used >= storage_limit:
            logger.warning("⚠️ Bucket is full. Pausing scraper...")
            send_discord_alert("⚠️ **Bucket Full**: The scraper is pausing because the bucket is full.")
            while storage_used >= storage_limit:
                time.sleep(60)  # Wait for 1 minute before checking again
                storage_used, storage_limit = check_bucket_storage(bucket_type="data")
            logger.info("✅ Bucket has available space. Resuming scraper...")
    except Exception as e:
        logger.error(f"❌ Error checking bucket storage: {e}")

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def start_kline_socket_with_backoff(symbol):
    """
    Start a kline WebSocket with retry logic.
    :param symbol: The trading pair symbol (e.g., "btcusdt").
    """
    twm.start_kline_socket(callback=handle_kline_message, symbol=symbol.lower(), interval="1m")

# Handle kline messages
def handle_kline_message(msg):
    """
    Callback for processing kline messages.
    """
    if msg['e'] != 'kline':
        return  # Not a kline message

    symbol = msg['s']
    k = msg['k']  # kline data payload

    # Debug: Print incoming kline data
    logger.debug(f"[DEBUG] OHLCV Update: {symbol} -> {k}")

    # Convert raw kline data
    kline_data = {
        'timestamp': pd.to_datetime(k['t'], unit='ms'),
        'open': float(k['o']),
        'high': float(k['h']),
        'low': float(k['l']),
        'close': float(k['c']),
        'volume': float(k['v'])
    }

    # Append to buffer
    if symbol not in ohlcv_buffer:
        ohlcv_buffer[symbol] = []

    ohlcv_buffer[symbol].append(kline_data)

    # Optional: Keep only the last 1000 rows to save memory
    if len(ohlcv_buffer[symbol]) > 1000:
        ohlcv_buffer[symbol] = ohlcv_buffer[symbol][-1000:]

    logger.info(f"[{symbol}] OHLCV updated: {kline_data}")

# Load symbols from environment variable or use default
symbols = os.getenv("SCRAPE_SYMBOLS", "btcusdt,ethusdt,solusdt").split(",")

# Main scraper logic
def main():
    try:
        logger.info("Starting Binance WebSocket OHLCV scraper...")
        send_discord_alert("✅ **Scraper Started**: The scraper is now running.")

        # Load OHLCV buffer from disk
        load_ohlcv_buffer_from_disk()

        # Start WebSocket manager
        twm.start()

        # Subscribe to kline sockets with retry logic
        for symbol in symbols:
            try:
                start_kline_socket_with_backoff(symbol)
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")
                send_discord_alert(f"❌ Failed to start WebSocket for {symbol}")

        # Run indefinitely and upload logs periodically
        while running:
            check_and_pause_if_bucket_full()  # Check if the bucket is full
            sleep_until_next_interval(interval_sec=300)  # Sync to the next interval
            save_all_to_parquet()  # Save OHLCV data periodically
            save_ohlcv_buffer_to_disk()  # Periodically save buffer to disk
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        send_discord_alert(f"❌ **Scraper Error**: {e}")
    finally:
        if running:  # Ensure graceful shutdown
            handle_sigint(None, None)

if __name__ == "__main__":
    main()



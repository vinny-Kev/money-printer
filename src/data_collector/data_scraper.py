"""
Binance Data Scraper with Local Storage
Updated to use centralized Discord notification system
"""
import os
import logging
import backoff
import requests
import pandas as pd
from binance import ThreadedWebsocketManager, Client
try:
    from src.data_collector.local_storage import (
        save_parquet_file,
        check_and_pause_if_storage_full as check_and_pause_if_bucket_full,
        check_storage_space as check_bucket_storage,
    )
except ImportError:
    # Fallback for standalone execution
    from local_storage import (
        save_parquet_file,
        check_and_pause_if_storage_full as check_and_pause_if_bucket_full,
        check_storage_space as check_bucket_storage,
    )
from datetime import datetime, timezone
from io import BytesIO
import signal
import sys
import time
import random
import json

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, DISCORD_WEBHOOK,
    CACHE_DIR, LOGS_DIR, DEFAULT_SYMBOLS, KLINE_INTERVAL,
    MAX_BUFFER_SIZE, SAVE_INTERVAL_SECONDS
)
from discord_notifications import send_scraper_notification

ohlcv_buffer = {}

# Create logs directory
os.makedirs(LOGS_DIR, exist_ok=True)

# Logger setup
log_filename = os.path.join(LOGS_DIR, f"log_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.txt")
logging.basicConfig(
    level=logging.DEBUG,  
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables - now using centralized config
if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    logger.error("Missing Binance API keys in environment variables.")
    sys.exit(1)

# Initialize Binance client and WebSocket manager
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

# Global running flag for graceful shutdown
running = True

def handle_sigint(signal_received, frame):
    """
    Handle SIGINT and SIGTERM signals for graceful shutdown.
    """
    global running
    logger.info("🛑 Gracefully stopping scraper...")
    send_scraper_notification("🛑 **Scraper Stopped**: The scraper is shutting down.")
    running = False

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# Discord alert function - now uses centralized notification system
def send_discord_alert(message):
    """
    Send an alert message to Discord using the centralized notification system.
    This function is kept for backward compatibility.
    """
    return send_scraper_notification(message)

# Save data to a single Parquet file per coin
def save_data_to_parquet(df, symbol):
    """
    Save OHLCV data for a symbol to a Parquet file, appending and deduplicating locally.
    """
    try:
        # Define the local file path
        filename = os.path.join(CACHE_DIR, f"{symbol}.parquet")

        # Handle deduplication and appending locally
        if os.path.exists(filename):
            existing_df = pd.read_parquet(filename)
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
            combined_df = pd.concat([existing_df, df])
        else:
            combined_df = df

        # Normalize timestamp and float precision
        combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
        float_cols = ["open", "high", "low", "close", "volume"]
        for col in float_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce").round(8)

        # Drop exact timestamp duplicates and sort
        combined_df = combined_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        
        # Save the deduplicated DataFrame locally using our new local storage
        save_parquet_file(combined_df, f"{symbol}.parquet", symbol)
        logger.info(f"✅ Saved deduplicated Parquet file locally: {filename}")

    except Exception as e:
        logger.error(f"❌ Failed to save Parquet file for {symbol}: {e}")

def save_all_to_parquet():
    """
    Save all OHLCV data to Parquet files using local storage.
    """
    global ohlcv_buffer  # Ensure we're modifying the global buffer
    try:
        for symbol in list(ohlcv_buffer.keys()):
            data = ohlcv_buffer[symbol]
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
        logger.info(f"✅ OHLCV buffer saved to disk at {CACHE_FILE} with {len(ohlcv_buffer)} symbols.")
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
                data = f.read().strip()
                if not data:  # Handle empty file
                    logger.warning(f"⚠️ OHLCV buffer file is empty. Initializing a new buffer.")
                    ohlcv_buffer = {}
                else:
                    ohlcv_buffer = json.loads(data)
            logger.info(f"✅ OHLCV buffer loaded from disk at {CACHE_FILE} with {len(ohlcv_buffer)} symbols.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"❌ Failed to load OHLCV buffer from disk: {e}")
            ohlcv_buffer = {}
    else:
        logger.info("No existing OHLCV buffer found on disk. Starting fresh.")
        ohlcv_buffer = {}

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

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def start_kline_socket_with_backoff(twm, symbol):
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def _start():
        twm.start_kline_socket(callback=handle_kline_message, symbol=symbol.lower(), interval="1m")
    _start()

# Handle kline messages
def handle_kline_message(msg):
    """
    Callback for processing kline messages.
    """
    if msg.get('e') != 'kline':
        logger.warning(f"Ignored non-kline message: {msg}")
        return  # Not a kline message

    global last_kline_time
    last_kline_time = time.time()  # Update the last kline message time only for valid messages

    symbol = msg['s']
    k = msg['k']  # kline data payload

    # Log the raw kline message
    logger.info(f"💓 Received kline for {symbol}: close={k['c']}")

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

    # Debug log for processed kline data
    logger.debug(f"✔️ Processed kline for {symbol}: {kline_data}")

    logger.info(f"[{symbol}] OHLCV updated: {kline_data}")

def get_top_100_trading_pairs():
    """
    Fetch the top 100 trading pairs by trading volume from Binance.
    :return: A list of trading pair symbols (e.g., ["BTCUSDT", "ETHUSDT", "SOLUSDT"]).
    """
    try:
        # Fetch 24-hour ticker price change statistics
        tickers = client.get_ticker()
        # Sort by quote volume (descending) and take the top 100
        top_pairs = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)[:100]
        # Filter for USDT pairs only
        usdt_pairs = [ticker['symbol'] for ticker in top_pairs if ticker['symbol'].endswith("USDT")]
        return usdt_pairs
    except Exception as e:
        logger.error(f"❌ Failed to fetch top trading pairs: {e}")
        return []

def main():
    global ohlcv_buffer  # Ensure we're modifying the global ohlcv_buffer
    ohlcv_buffer = ohlcv_buffer if isinstance(ohlcv_buffer, dict) else {}

    # ⏱️ Track the last kline message time
    global last_kline_time
    last_kline_time = time.time()

    # Fetch the top 100 trading pairs
    symbols = get_top_100_trading_pairs()

    if not symbols:
        logger.error("❌ No trading pairs to scrape. Exiting.")
        sys.exit(1)

    logger.info(f"✅ Scraping the following trading pairs: {', '.join(symbols)}")
    send_scraper_notification(f"🚀 **Data Scraper Started**: Now monitoring {len(symbols)} trading pairs")

    # Initialize WebSocket manager
    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

    try:
        # Load OHLCV buffer from disk
        load_ohlcv_buffer_from_disk()

        # Initialize buffer if empty
        if not ohlcv_buffer:
            logger.warning("⚠️ OHLCV buffer is empty. Initializing a new buffer.")
            ohlcv_buffer = {}

        # Start WebSocket manager if not already running
        if not twm.is_alive():
            twm.start()
            logger.info("✅ WebSocket manager started.")

        # Subscribe to kline sockets AFTER manager is running
        for symbol in symbols:
            try:
                start_kline_socket_with_backoff(twm, symbol)
                logger.info(f"🧩 Started kline socket for {symbol}")
                time.sleep(0.2)  # ✅ Throttle subscriptions to 5 per second
            except Exception as e:
                logger.error(f"❌ Failed to start WebSocket for {symbol}: {e}")
                send_scraper_notification(f"❌ Failed to start WebSocket for {symbol}")

        # 👁️ Watchdog loop
        while running:
            # Check if no kline messages have been received in the last 60 seconds
            if time.time() - last_kline_time > 60:
                logger.warning("⚠️ No kline messages received in the last 60 seconds.")

            # Perform periodic tasks
            check_and_pause_if_bucket_full()  # Check if the bucket is full
            sleep_until_next_interval(interval_sec=300)  # Sync to the next interval
            save_all_to_parquet()  # Save OHLCV data periodically
            save_ohlcv_buffer_to_disk()  # Periodically save buffer to disk
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        send_scraper_notification(f"❌ **Scraper Error**: {e}")
    finally:
        logger.info("🔻 Shutting down gracefully.")
        if twm:
            try:
                twm.stop()
                logger.info("✅ WebSocket manager stopped.")
            except Exception as e:
                logger.error(f"❌ Failed to stop WebSocket manager: {e}")

if __name__ == "__main__":
    main()

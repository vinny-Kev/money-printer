import os
import json
import pickle
import binance
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance.client import Client
import requests
import time
import subprocess
import csv
import logging
import sqlite3
from datetime import datetime
from binance.exceptions import BinanceAPIException
from functools import wraps
import random
import traceback
import sys
import warnings
from warnings import simplefilter
from binance import ThreadedWebsocketManager
from itertools import cycle

print("Starting live trader...")

TP_PROFIT_MARGIN = 0.05     # Aim for 5% net profit
SL_PERCENT = 0.01           # 1% stop-loss
TP_SL_CHECK_DURATION = 1800  # 30 mins max
TP_SL_CHECK_INTERVAL = 30    # check every 30s
TRADING_RECIPTS_PATH = os.path.join(os.path.dirname(__file__), "trading_receipts.csv")
REFINEMENT_LOG_PATH = os.path.join(os.path.dirname(__file__), "refinement_log.csv")

# --- Logging Setup ---
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.join(LOGS_DIR, "trades"), exist_ok=True)
os.makedirs(os.path.join(LOGS_DIR, "market_data"), exist_ok=True)
os.makedirs(os.path.join(LOGS_DIR, "transactions"), exist_ok=True)

# Setup logging to both file and console
log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# File handler
file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "live_trader.log"))
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Setup logger
logger = logging.getLogger("live_trader")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
results = {}
# Keep original print for debugging
original_print = print


session_trades = []
class TradeLogger:
    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database and create the trades table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        reason TEXT,
                        outcome INTEGER DEFAULT NULL
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            safe_print(f"Error initializing database: {e}")

    def log_trade(self, symbol, action, price, quantity, reason, outcome=None):
        """Log a trade into the database."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trades (timestamp, symbol, action, price, quantity, reason, outcome)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, symbol, action, price, quantity, reason, outcome))
                conn.commit()
            safe_print(f"Logged trade to database: {symbol} {action} {quantity} @ {price}")
        except sqlite3.Error as e:
            safe_print(f"Error logging trade to database: {e}")

    def fetch_trades(self, limit=100):
        """Fetch the most recent trades from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM trades
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                trades = cursor.fetchall()
                return trades
        except sqlite3.Error as e:
            safe_print(f"Error fetching trades: {e}")
            return []

    def delete_trade(self, trade_id):
        """Delete a trade from the database by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM trades
                    WHERE id = ?
                """, (trade_id,))
                conn.commit()
            safe_print(f"Deleted trade with ID: {trade_id}")
        except sqlite3.Error as e:
            safe_print(f"Error deleting trade: {e}")

# Initialize trade_logger
trade_logger = TradeLogger(os.path.join(LOGS_DIR, "trades.db"))

# Replace the `safe_print` function with direct logger calls
def safe_print(msg, overwrite=False):
    """
    Print a message to the terminal. If overwrite=True, it overwrites the previous line.
    :param msg: The message to print.
    :param overwrite: Whether to overwrite the previous line.
    """
    try:
        if overwrite:
            sys.stdout.write(f"\r{msg}{' ' * (os.get_terminal_size().columns - len(msg))}")  # Clear remaining text
            sys.stdout.flush()
        else:
            logger.info(str(msg))  # Replace with logger.info
    except Exception as e:
        logger.error(f"Logging failed: {e}")
        logger.error(msg)

# --- Exponential Backoff Decorator ---
def binance_retry(max_retries=5, base_delay=2, jitter=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except BinanceAPIException as e:
                    if e.status_code == 429:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
                        safe_print(f"Rate limit hit. Retrying in {delay:.1f}s (attempt {attempt+1})")
                        time.sleep(delay)
                    else:
                        safe_print(f"Binance API error: {e}")
                        raise
                except Exception as e:
                    safe_print(f"Error in {func.__name__}: {e}")
                    safe_print(f"Traceback: {traceback.format_exc()}")
                    raise
            safe_print(f"Max retries exceeded for {func.__name__}")
            return None
        return wrapper
    return decorator

# --- TradeExecutor Abstraction ---
class TradeExecutor:
    def __init__(self, client, live_trading):
        self.client = client
        self.live_trading = live_trading

    @binance_retry()
    def market_buy(self, symbol, quantity):
        try:
            if self.live_trading:
                order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
                safe_print(f"Executed LIVE BUY: {order}")
                return order
            else:
                safe_print(f"[DRY RUN] Would BUY {symbol} qty {quantity}")
                return {"symbol": symbol, "side": "BUY", "quantity": quantity, "dry_run": True}
        except Exception as e:
            safe_print(f"Error in market_buy: {e}")
            safe_print(f"Traceback: {traceback.format_exc()}")
            raise

    @binance_retry()
    def market_sell(self, symbol, quantity):
        try:
            if self.live_trading:
                order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
                safe_print(f"Executed LIVE SELL: {order}")
                return order
            else:
                safe_print(f"[DRY RUN] Would SELL {symbol} qty {quantity}")
                return {"symbol": symbol, "side": "SELL", "quantity": quantity, "dry_run": True}
        except Exception as e:
            safe_print(f"Error in market_sell: {e}")
            safe_print(f"Traceback: {traceback.format_exc()}")
            raise

    def place_tp_sl_orders(self, symbol, quantity, take_profit, stop_loss):
        if self.live_trading:
            safe_print(f"[LIVE] Would place TP: {take_profit:.2f}, SL: {stop_loss:.2f} for {symbol}")
            # Actual TP/SL logic can be implemented here.
        else:
            safe_print(f"[DRY RUN] Would place TP: {take_profit:.2f}, SL: {stop_loss:.2f} for {symbol}")

# --- Model/Scaler Loader (single entrypoint) ---
def load_model_and_scaler():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        safe_print("Model loaded successfully")
        
        scaler_path = MODEL_PATH.replace(".pkl", "_scaler.pkl")
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            safe_print("Scaler loaded successfully")
        else:
            safe_print("No scaler file found, proceeding without scaling")
        
        return model, scaler
    except Exception as e:
        safe_print(f"Failed to load model/scaler: {e}")
        safe_print(f"Traceback: {traceback.format_exc()}")
        raise

# --- Trade Logging ---
def log_transaction(symbol, action, price, quantity, timestamp, reason):
    """
    Log trade details to a Parquet file for transaction history.
    :param symbol: The trading symbol (e.g., BTCUSDT).
    :param action: The trade action (BUY/SELL).
    :param price: The price at which the trade was executed.
    :param quantity: The quantity traded.
    :param timestamp: The timestamp of the trade.
    :param reason: The reason for the trade.
    """
    try:
        filename = os.path.join(LOGS_DIR, "transactions.parquet")
        row = {
            "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": quantity,
            "reason": reason
        }
        # Load existing data if the file exists
        if os.path.exists(filename):
            existing_data = pd.read_parquet(filename)
            new_data = pd.DataFrame([row])
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_data = pd.DataFrame([row])

        # Save to Parquet
        combined_data.to_parquet(filename, index=False)
        safe_print(f"Logged transaction to Parquet: {symbol} {action} {quantity} @ {price}")
    except Exception as e:
        safe_print(f"Failed to log transaction: {e}")

def log_trade_csv(symbol, action, price, quantity, reason):
    """
    Log trade details to a CSV file for taxation purposes.
    :param symbol: The trading symbol (e.g., BTCUSDT).
    :param action: The trade action (BUY/SELL).
    :param price: The price at which the trade was executed.
    :param quantity: The quantity traded.
    :param reason: The reason for the trade.
    """
    try:
        filename = os.path.join(LOGS_DIR, "trades", "tax_trades.csv")
        fieldnames = ["timestamp", "symbol", "action", "price", "quantity", "reason"]
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": quantity,
            "reason": reason
        }
        write_header = not os.path.exists(filename)
        with open(filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        safe_print(f"Logged trade to tax CSV: {symbol} {action} {quantity} @ {price}")
    except Exception as e:
        safe_print(f"Error logging trade to tax CSV: {e}")

# --- Config/Constants ---
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(CURRENT_DIR, ".env")
    load_dotenv(dotenv_path)

    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
    BINANCE_API_KEY_TESTNET = os.getenv("BINANCE_API_KEY_TESTNET")
    BINANCE_SECRET_KEY_TESTNET = os.getenv("BINANCE_SECRET_KEY_TESTNET")
    
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        raise ValueError("Missing Binance API credentials in .env file")

    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
    DATA_DIR = os.path.join(BASE_DIR, "data/scraped_data")
    MODEL_PATH = os.path.join(BASE_DIR, "data/models/trained_model.pkl")
    FEATURE_PATH = os.path.join(BASE_DIR, "data/models/important_features.json")
    PAPER_TRADER_PATH = os.path.join(BASE_DIR, "src/paper_trading/paper_trader.py")

    safe_print(f"MODEL_PATH: {MODEL_PATH}")
    safe_print(f"FEATURE_PATH: {FEATURE_PATH}")
    
    # Verify critical files exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(f"Features file not found: {FEATURE_PATH}")

except Exception as e:
    safe_print(f"Configuration error: {e}")
    safe_print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)

# Trading constants
PLATFORM_FEE = 0.001
SLIPPAGE_RATE = 0.001
GAS_FEE = 0.0
CONFIDENCE_THRESHOLD = 0.5
INITIAL_CAPITAL = 500
VOLATILITY_SPIKE_THRESHOLD = 0.05
COOLDOWN_MINUTES = 30
TRAILING_STOP_STEP = 0.01
MIN_USDT_BALANCE = 3
LIVE_TRADING = False

# Initialize Binance client with error handling
client = None
try:
    client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
    safe_print("Binance client initialized successfully")
except Exception as e:
    safe_print(f"Failed to initialize Binance client: {e}")
    safe_print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)

# Load important features
important_features = None
try:
    with open(FEATURE_PATH, "r") as f:
        important_features = json.load(f)
    safe_print(f"Loaded {len(important_features)} important features")
except Exception as e:
    safe_print(f"Failed to load important features: {e}")
    safe_print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)

# Ensure `model` and `scaler` are initialized globally
model, scaler = None, None
try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    safe_print(f"Failed to load model/scaler: {e}")
    sys.exit(1)

def get_trade_fraction(balance):
    if balance < 100:
        return 0.25
    elif balance < 1000:
        return 0.15
    elif balance < 10000:
        return 0.10
    else:
        return 0.05

def get_required_gain(trade_amount):
    total_fees = (PLATFORM_FEE * 2 * trade_amount) + GAS_FEE
    slippage_loss = SLIPPAGE_RATE * 2
    return (total_fees / trade_amount) + slippage_loss

def get_binance_balance(asset="USDT"):
    try:
        balance_info = client.get_asset_balance(asset=asset)
        if balance_info is None:
            safe_print(f"No balance info returned for {asset}")
            return None
        return float(balance_info['free'])
    except Exception as e:
        safe_print(f"Error fetching Binance balance for {asset}: {e}")
        safe_print(f"Traceback: {traceback.format_exc()}")
        return None

def run_paper_trader():
    try:
        safe_print("Running paper trader for test...")
        if not os.path.exists(PAPER_TRADER_PATH):
            safe_print(f"Paper trader script not found: {PAPER_TRADER_PATH}")
            return
        subprocess.run(["python", PAPER_TRADER_PATH], check=True)
    except Exception as e:
        safe_print(f"Error running paper trader: {e}")

def get_latest_price(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol.upper()+"USDT")
        return float(ticker['price'])
    except Exception as e:
        safe_print(f"Error fetching price for {symbol}: {e}")
        return None

@binance_retry(max_retries=5, base_delay=2, jitter=1)
def print_binance_balances(min_balance=0.0001):
    try:
        safe_print("Fetching Binance balances...")
        account_info = client.get_account()  # Removed timeout parameter
        balances = account_info.get('balances', [])
        
        safe_print("--- Your Binance Wallet Balances ---")
        for asset in balances:
            free = float(asset['free'])
            locked = float(asset['locked'])
            total = free + locked
            if total >= min_balance:
                safe_print(f"{asset['asset']}: {total:.8f} (Free: {free:.8f}, Locked: {locked:.8f})")
        safe_print("------------------------------------")
    except requests.exceptions.ReadTimeout:
        safe_print("Timeout fetching Binance account balances. Retrying...")
    except Exception as e:
        safe_print(f"Error fetching Binance balances: {e}")
        safe_print(f"Traceback: {traceback.format_exc()}")

@binance_retry(max_retries=5, base_delay=2, jitter=1)
def fetch_latest_binance_ohlcv(symbol):
    """Fetch the latest OHLCV data from the WebSocket stream."""
    try:
        if symbol in live_ohlcv_data:
            return live_ohlcv_data[symbol]
        else:
            logger.warning(f"No live OHLCV data available for {symbol}.")
            return None
    except Exception as e:
        logger.error(f"Error fetching live OHLCV data for {symbol}: {e}")
        return None

def fetch_latest_binance_ohlcv_websocket(symbol, interval="30m"):
    """Fetch OHLCV data using Binance WebSocket streams."""
    try:
        twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
        twm.start()

        ohlcv_data = {}

        def handle_socket_message(msg):
            ohlcv_data.update({
                "timestamp": pd.to_datetime(msg['k']['t'], unit="ms"),
                "open": float(msg['k']['o']),
                "high": float(msg['k']['h']),
                "low": float(msg['k']['l']),
                "close": float(msg['k']['c']),
                "volume": float(msg['k']['v']),
            })
            twm.stop()  # Stop WebSocket after receiving data

        twm.start_kline_socket(callback=handle_socket_message, symbol=symbol.upper() + "USDT", interval=interval)
        twm.join()  # Wait for WebSocket to finish

        return ohlcv_data if ohlcv_data else None
    except Exception as e:
        safe_print(f"Error fetching live OHLCV via WebSocket for {symbol}: {e}")
        return None

def calculate_quantity(trade_amount, price, symbol):
    try:
        # Fetch symbol info to get the minQty filter
        symbol_info = client.get_symbol_info(symbol)
        if not symbol_info or 'filters' not in symbol_info:
            safe_print(f"Symbol info not found or invalid for {symbol}")
            return None

        # Extract minQty from filters
        min_qty_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not min_qty_filter or 'minQty' not in min_qty_filter:
            safe_print(f"minQty filter not found for {symbol}")
            return None

        min_qty = float(min_qty_filter['minQty'])
        qty = trade_amount / price
        return max(round(qty, 6), min_qty)  # Ensure quantity meets minQty
    except Exception as e:
        safe_print(f"Error calculating quantity for {symbol}: {e}")
        return None

def get_prediction_confidence(model, features):
    try:
        # Ensure features are passed as a DataFrame with valid column names
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features, columns=important_features)

        probas = model.predict_proba(features)[0]
        confidence = probas[1] if probas[1] > 0.5 else 1 - probas[1]
        prediction = np.argmax(probas)
        return prediction, confidence, probas
    except Exception as e:
        safe_print(f"Error getting prediction confidence: {e}")
        safe_print(f"Traceback: {traceback.format_exc()}")
        return 0, 0.0, [0.5, 0.5]

def construct_live_row(ohlcv, features):
    try:
        row = {}
        for col in features:
            # Check if the column exists in ohlcv; if not, set it to 0
            row[col] = ohlcv.get(col, 0)
        return row
    except Exception as e:
        safe_print(f"Error constructing live row: {e}")
        return {col: 0 for col in features}

def get_testnet_client():
    try:
        api_key = os.getenv("BINANCE_API_KEY_TESTNET")
        api_secret = os.getenv("BINANCE_SECRET_KEY_TESTNET")
        if not api_key or not api_secret:
            raise ValueError("Missing testnet API keys in .env file")
        testnet_client = Client(api_key, api_secret)
        testnet_client.API_URL = 'https://testnet.binance.vision/api'  # Correct testnet URL
        safe_print("Testnet client initialized successfully")
        return testnet_client
    except Exception as e:
        safe_print(f"Error creating testnet client: {e}")
        raise

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"{category.__name__}: {message} [{filename}:{lineno}]")

warnings.showwarning = custom_warning_handler
simplefilter("always")  # Still trigger them for logging

def analyze_symbols(symbol):
    try:
        balance = get_binance_balance()
        if balance is None or balance < MIN_USDT_BALANCE:
            logger.warning("Insufficient balance for trading. Exiting.")
            return None

        full_symbol = symbol.upper() + "USDT"
        symbol_info = client.get_symbol_info(full_symbol)
        if symbol_info is None:
            logger.warning(f"Skipping {full_symbol} — Not a valid Binance symbol.")
            return None

        trade_amount = balance * get_trade_fraction(balance)
        if LIVE_TRADING and balance - trade_amount < MIN_USDT_BALANCE:
            logger.warning(f"Skipping {symbol} — Would breach minimum capital protection limit.")
            return None

        ohlcv = fetch_latest_binance_ohlcv(symbol)
        if ohlcv is None:
            logger.warning(f"Skipping {symbol} — No live OHLCV data.")
            return None

        live_row = construct_live_row(ohlcv, important_features)
        live_df = pd.DataFrame([live_row])
        features = live_df[important_features].select_dtypes(include=[np.number]).fillna(0)

        if scaler:
            features = scaler.transform(features)
        else:
            features = features.values

        prediction, confidence, probas = get_prediction_confidence(model, features)
        expected_gain = max((probas[1] - 0.5) * 0.2, 0.01)
        required_gain = get_required_gain(trade_amount)

        logger.info(f"Analyzed {symbol}: Confidence={confidence:.2f}, Expected Gain={expected_gain:.2f}, Required Gain={required_gain:.2f}")

        if confidence < CONFIDENCE_THRESHOLD or abs(expected_gain) < required_gain:
            logger.warning(f"Skipping {symbol} — Low confidence or expected gain below threshold")
            return None

        return full_symbol, {
            "prediction": prediction,
            "confidence": confidence,
            "expected_gain": expected_gain,
            "required_gain": required_gain,
        }
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {e}")
        return None


def get_top_symbols(limit=100):
    try:
        tickers = client.get_ticker()
        valid_symbols = [
            ticker['symbol'] for ticker in tickers
            if ticker['symbol'].endswith("USDT") and client.get_symbol_info(ticker['symbol']) is not None
        ]
        safe_print(f"Fetched top symbols: {valid_symbols[:limit]}")
        return valid_symbols[:limit]
    except Exception as e:
        safe_print(f"Error fetching top symbols: {e}")
        return []

# --- Log Refinement Data ---
def log_refinement_data(symbol, features, prediction, confidence, outcome, timestamp):
    try:
        row = {feature: features[feature] for feature in important_features}
        row.update({
            "timestamp": timestamp,
            "symbol": symbol,
            "prediction": prediction,
            "confidence": confidence,
            "outcome": outcome
        })
        write_header = not os.path.exists(REFINEMENT_LOG_PATH)
        with open(REFINEMENT_LOG_PATH, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        safe_print(f"Logged refinement data for {symbol}.")
    except Exception as e:
        safe_print(f"Failed to log refinement data: {e}")

# --- Save Trading Receipts ---
def save_trading_receipts(session_trades):
    try:
        write_header = not os.path.exists(TRADING_RECIPTS_PATH)
        with open(TRADING_RECIPTS_PATH, "a", newline="") as csvfile:
            fieldnames = ["timestamp", "symbol", "action", "price", "quantity", "confidence", "gain", "take_profit", "stop_loss", "outcome"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for trade in session_trades:
                writer.writerow(trade)
        safe_print("[+] Trading receipts saved successfully.")
    except Exception as e:
        safe_print(f"[!] Failed to save trading receipts: {e}")

# --- Check Refinement Data Size ---
def check_refinement_data_size(refinement_csv, min_rows=50):
    try:
        if not os.path.exists(refinement_csv):
            safe_print(f"[!] Refinement data file not found: {refinement_csv}")
            return False
        ref_df = pd.read_csv(refinement_csv)
        return len(ref_df) >= min_rows
    except Exception as e:
        safe_print(f"[!] Failed to check refinement data size: {e}")
        return False

# --- Prompt User to Run Trainer ---
def prompt_run_trainer():
    try:
        safe_print("Checking if enough refinement data has been accumulated...")
        if check_refinement_data_size(REFINEMENT_LOG_PATH, min_rows=50):
            safe_print("[+] Enough refinement data accumulated.")
            user_input = input("Do you want to run the trainer to update the model? (y/n): ").strip().lower()
            if user_input == "y":
                safe_print("Running trainer script...")
                subprocess.run(["python", "src/common/trainer.py"], check=True)
                safe_print("[+] Trainer script completed successfully.")
            else:
                safe_print("Trainer script skipped.")
        else:
            safe_print("[!] Not enough refinement data accumulated to run the trainer.")
    except Exception as e:
        safe_print(f"[!] Failed to prompt user for trainer: {e}")

# --- Spinner Function ---
def spinner(message, duration=5):
    """Display a spinner in the terminal."""
    spinner_cycle = cycle(['|', '/', '-', '\\'])
    end_time = time.time() + duration
    while time.time() < end_time:
        sys.stdout.write(f"\r{message} {next(spinner_cycle)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * len(message) + "\r")  # Clear spinner
    sys.stdout.flush()

# --- Trade Outcome Labeling ---
def label_trade_outcome(entry_price, exit_price, threshold=0.01):
    """
    Label the trade outcome as a win (1) or loss (0).
    :param entry_price: The price at which the trade was entered.
    :param exit_price: The price at which the trade was exited.
    :param threshold: Minimum profit percentage to consider the trade a win.
    :return: 1 if profit >= threshold, otherwise 0.
    """
    try:
        profit = (exit_price - entry_price) / entry_price
        return 1 if profit >= threshold else 0
    except Exception as e:
        safe_print(f"Error labeling trade outcome: {e}")
        return 0

# --- Initialize Logs ---
def initialize_logs():
    try:
        # Create refinement log file if it doesn't exist
        if not os.path.exists(REFINEMENT_LOG_PATH):
            with open(REFINEMENT_LOG_PATH, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "symbol", "prediction", "confidence", "outcome"])
                writer.writeheader()
        # Create trading receipts file if it doesn't exist
        if not os.path.exists(TRADING_RECIPTS_PATH):
            with open(TRADING_RECIPTS_PATH, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "symbol", "action", "price", "quantity", "confidence", "gain", "take_profit", "stop_loss", "outcome"])
                writer.writeheader()
    except Exception as e:
        safe_print(f"Error initializing logs: {e}")

# --- PnL Calculation ---
def calculate_pnl():
    """
    Calculate and display PnL from transaction history saved in Parquet.
    """
    try:
        filename = os.path.join(LOGS_DIR, "transactions.parquet")
        if not os.path.exists(filename):
            safe_print("No transaction history found.")
            return

        transactions = pd.read_parquet(filename)
        if transactions.empty:
            safe_print("Transaction history is empty.")
            return

        # Calculate PnL
        transactions["pnl"] = transactions.apply(
            lambda row: row["quantity"] * row["price"] if row["action"] == "SELL" else -row["quantity"] * row["price"],
            axis=1
        )
        total_pnl = transactions["pnl"].sum()

        # Display PnL table
        safe_print("\n--- PnL Summary ---")
        safe_print(transactions[["timestamp", "symbol", "action", "price", "quantity", "pnl"]].to_string(index=False))
        safe_print(f"\nTotal PnL: {total_pnl:.2f} USDT")
    except Exception as e:
        safe_print(f"Failed to calculate PnL: {e}")

def dry_trader(session_trades):
    """
    Simulate selling all coins in the session trades.
    """
    try:
        for trade in session_trades:
            sell_price = trade["take_profit"]  # Simulate TP hit
            safe_print(f"[DRY RUN] SELL {trade['symbol']} qty {trade['quantity']} @ {sell_price:.2f}")
            trade["action"] = "SELL"
            trade["price"] = sell_price
            trade["pnl"] = (sell_price - trade["price"]) * trade["quantity"]
    except Exception as e:
        safe_print(f"Error in dry trader: {e}")

def validate_symbol(symbol, results):
    """Check if the symbol is valid on Binance."""
    try:
        symbol_info = client.get_symbol_info(symbol)
        return symbol_info is not None
    except Exception as e:
        safe_print(f"Error validating symbol {symbol}: {e}")
        return False

virtual_balance = binance  # Respect testnet budget
held_symbols = set()  # Track symbols already held

for full_symbol, data in results.items(): 
    if full_symbol in held_symbols:
        safe_print(f"Skipping {full_symbol} — already holding.")
        continue

    # Validate symbol
    if not validate_symbol(full_symbol):
        safe_print(f"Skipping {full_symbol} — Invalid Binance symbol.")
        continue

    trade_amount = data["trade_amount"]
    if trade_amount > virtual_balance:
        safe_print(f"Skipping {full_symbol} — not enough virtual budget (${virtual_balance:.2f} left).")
        continue

    try:
        ohlcv = fetch_latest_binance_ohlcv(full_symbol)
        if ohlcv is None:
            safe_print(f"Skipping {full_symbol} — No OHLCV data available.")
            continue

        price = ohlcv["close"]
        quantity = calculate_quantity(trade_amount, price, full_symbol)

        if quantity is None or quantity * price < MIN_USDT_BALANCE:
            safe_print(f"Skipping {full_symbol} — Trade amount below minimum threshold.")
            continue

        timestamp = time.time()

        # Simulate BUY
        log_transaction(full_symbol, "BUY", price, quantity, timestamp, reason="Confidence and gain threshold met")
        safe_print(f"[{full_symbol}] BUY {quantity:.5f} @ ${price:.2f} | Confidence: {data['confidence']:.2f} | Gain: {data['expected_gain']:.3f}")
        safe_print("-" * 50)

        # Update virtual balance and held symbols
        virtual_balance -= trade_amount
        held_symbols.add(full_symbol)

        # Simulate TP/SL
        take_profit = price * (1 + TP_PROFIT_MARGIN)
        stop_loss = price * (1 - SL_PERCENT)
        safe_print(f"[DRY RUN] Would place TP: ${take_profit:.2f}, SL: ${stop_loss:.2f} for {full_symbol}")

        # Simulate trade outcome
        sell_price = take_profit  # Simulate TP hit
        outcome = label_trade_outcome(price, sell_price)

        session_trades.append({
            "timestamp": timestamp,
            "symbol": full_symbol,
            "action": "BUY",
            "price": price,
            "quantity": quantity,
            "confidence": data["confidence"],
            "gain": TP_PROFIT_MARGIN,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "outcome": outcome
        })

    except Exception as e:
        safe_print(f"Trade error for {full_symbol}: {e}")



# --- Main Trading Logic ---
def main():
    try:
        logger.info("Main function started")
        global client, LIVE_TRADING

        # Test client connection
        try:
            logger.info("Testing client connection...")
            print_binance_balances()
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return

        # --- Trading mode selection ---
        balance = get_binance_balance()
        testnet_budget = None

        if balance is None or balance == 0:
            logger.info("No live balance detected. Automatically switching to TESTNET mode.")
            mode = "d"
        else:
            mode = input("Select trading mode (d: testnet, l: live, q: quit): ").strip().lower()

        if mode == "q":
            logger.info("Aborting trading bot.")
            return
        elif mode == "d":
            logger.info("TESTNET trading mode selected.")
            try:
                client = get_testnet_client()
                LIVE_TRADING = False
                testnet_budget = float(input("Enter your TESTNET budget in USDT (default 500): ") or "500")
                balance = testnet_budget
                logger.info(f"Using TESTNET budget: ${balance:.2f}")
            except Exception as e:
                logger.error(f"Failed to initialize testnet client: {e}")
                return
        elif mode == "l":
            logger.info("LIVE trading mode selected.")
            run_paper_trader()
            input("Review paper trader results above. Press Enter to continue to LIVE trading...")
            LIVE_TRADING = True
            balance = get_binance_balance()
            if balance is None or balance < MIN_USDT_BALANCE:
                logger.warning("Insufficient live balance for trading. Exiting.")
                return
        else:
            logger.warning("Invalid selection. Exiting.")
            return

        # Initialize WebSocket manager and subscribe to OHLCV streams
        start_websocket_stream()
        SYMBOLS = get_top_symbols(limit=100)
        subscribe_to_ohlcv_streams(SYMBOLS)

        # --- Analyze and Select Best Coin ---
        best_coin = None
        best_confidence = 0
        best_data = None

        for symbol in SYMBOLS:
            logger.info(f"Analyzing symbol: {symbol}")
            analysis = analyze_symbols(symbol)
            if analysis:
                full_symbol, data = analysis
                logger.info(f"Analysis result for {symbol}: {data}")
                if data["confidence"] > best_confidence:
                    best_coin = full_symbol
                    best_confidence = data["confidence"]
                    best_data = data

        if not best_coin:
            logger.info("No suitable coin found for trading.")
            return

        logger.info(f"Best coin selected: {best_coin} with confidence {best_confidence:.2f}")

        # --- Execute Trade ---
        try:
            trade_amount = balance * get_trade_fraction(balance)
            ohlcv = fetch_latest_binance_ohlcv(best_coin)
            if ohlcv is None:
                logger.warning(f"Skipping {best_coin} — No OHLCV data available.")
                return

            price = ohlcv["close"]
            quantity = calculate_quantity(trade_amount, price, best_coin)

            if quantity is None or quantity * price < MIN_USDT_BALANCE:
                logger.warning(f"Skipping {best_coin} — Trade amount below minimum threshold.")
                return

            timestamp = time.time()

            # Place buy order
            if LIVE_TRADING:
                logger.info(f"Placing LIVE BUY order for {best_coin}...")
                trade_executor.market_buy(best_coin, quantity)
            else:
                logger.info(f"[DRY RUN] Would BUY {best_coin} qty {quantity}")
                log_transaction(best_coin, "BUY", price, quantity, timestamp, reason="Confidence and gain threshold met")
            logger.info(f"[{best_coin}] BUY {quantity:.5f} @ ${price:.2f} | Confidence: {best_confidence:.2f}")

            # Set TP/SL
            take_profit = price * (1 + TP_PROFIT_MARGIN)
            stop_loss = price * (1 - SL_PERCENT)
            logger.info(f"Setting TP: ${take_profit:.2f}, SL: ${stop_loss:.2f} for {best_coin}")

            if LIVE_TRADING:
                trade_executor.place_tp_sl_orders(best_coin, quantity, take_profit, stop_loss)

            # Wait for trade completion
            logger.info(f"Monitoring trade for {best_coin}...")
            while True:
                current_price = get_latest_price(best_coin)
                if current_price is None:
                    logger.warning(f"Error fetching price for {best_coin}. Retrying...")
                    time.sleep(TP_SL_CHECK_INTERVAL)
                    continue

                if current_price >= take_profit:
                    logger.info(f"Take profit hit for {best_coin} at ${current_price:.2f}. Exiting trade.")
                    break
                elif current_price <= stop_loss:
                    logger.info(f"Stop loss hit for {best_coin} at ${current_price:.2f}. Exiting trade.")
                    break

                time.sleep(TP_SL_CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Error executing trade for {best_coin}: {e}")

    except Exception as e:
        logger.error(f"Error in main trading loop: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("Live trader execution completed.")

# Global WebSocket manager and OHLCV data storage
twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
live_ohlcv_data = {}  # Dictionary to store the latest OHLCV data for each symbol

def start_websocket_stream():
    """Start the WebSocket manager."""
    try:
        twm.start()
        logger.info("WebSocket manager started successfully.")
    except Exception as e:
        logger.error(f"Failed to start WebSocket manager: {e}")

if __name__ == "__main__":
    main()
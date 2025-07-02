"""
Central configuration module for the trading system.
This replaces scattered configuration across different modules.
"""
import os
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available - use os.environ directly
    pass

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = Path(__file__).parent

# Data directories
DATA_ROOT = PROJECT_ROOT / "data"
SCRAPED_DATA_DIR = DATA_ROOT / "scraped_data"
PARQUET_DATA_DIR = SCRAPED_DATA_DIR / "parquet_files"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model directories
MODELS_DIR = DATA_ROOT / "models"
RF_MODEL_DIR = MODELS_DIR / "random_forest"
XGB_MODEL_DIR = MODELS_DIR / "xgboost"
LSTM_MODEL_DIR = MODELS_DIR / "lstm"

# Ensure directories exist
for directory in [DATA_ROOT, SCRAPED_DATA_DIR, PARQUET_DATA_DIR, CACHE_DIR, LOGS_DIR, 
                  MODELS_DIR, RF_MODEL_DIR, XGB_MODEL_DIR, LSTM_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BINANCE_API_KEY_TESTNET = os.getenv("BINANCE_API_KEY_TESTNET")
BINANCE_SECRET_KEY_TESTNET = os.getenv("BINANCE_SECRET_KEY_TESTNET")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "False").lower() == "true"

# Discord Configuration
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DISCORD_WEBHOOK_DATA_SCRAPER = os.getenv("DISCORD_WEBHOOK_DATA_SCRAPER")
DISCORD_WEBHOOK_TRAINERS = os.getenv("DISCORD_WEBHOOK_TRAINERS")
DISCORD_WEBHOOK_TRADERS = os.getenv("DISCORD_WEBHOOK_TRADERS")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

# Railway Configuration
RAILWAY_API_TOKEN = os.getenv("RAILWAY_API_TOKEN")
RAILWAY_PROJECT_ID = os.getenv("RAILWAY_PROJECT_ID")
RAILWAY_MAX_USAGE_HOURS = float(os.getenv("RAILWAY_MAX_USAGE_HOURS", "450"))  # Monthly limit
RAILWAY_WARNING_HOURS = float(os.getenv("RAILWAY_WARNING_HOURS", "400"))     # Warning threshold
RAILWAY_CHECK_INTERVAL = int(os.getenv("RAILWAY_CHECK_INTERVAL", "5"))       # Check every 5 minutes

# Google Drive Configuration
USE_GOOGLE_DRIVE = os.getenv("USE_GOOGLE_DRIVE", "False").lower() == "true"
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")  # Target folder for sync
SECRETS_DIR = PROJECT_ROOT / "secrets"
DRIVE_CREDENTIALS_PATH = SECRETS_DIR / "credentials.json"
DRIVE_TOKEN_PATH = SECRETS_DIR / "token.json"

# Ensure secrets directory exists
SECRETS_DIR.mkdir(parents=True, exist_ok=True)

# Data Collection Settings
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT"]
KLINE_INTERVAL = "1m"
MAX_BUFFER_SIZE = 1000
SAVE_INTERVAL_SECONDS = 60  # Save every 60 seconds for testing (was 300)

# Storage Settings
MAX_LOCAL_STORAGE_GB = 10
CLEANUP_OLD_FILES_DAYS = 30
MAX_FILES_PER_SYMBOL = 100

# Model Training Settings
MIN_TRAINING_ROWS = 500
TRAIN_TEST_SPLIT = 0.7
RANDOM_STATE = 42

# Trading Settings
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "True").lower() == "true"  # Master trading switch - ENABLED
PAPER_TRADING = False  # Set to False for live trading - USING LIVE BINANCE API
live_trading = not PAPER_TRADING  # For compatibility with trade_runner.py
DEFAULT_TRADE_AMOUNT = 100  # USDT
STOP_LOSS_PERCENT = 2.0
TAKE_PROFIT_PERCENT = 4.0

# PRODUCTION SAFETY SETTINGS
MAX_TRADES_PER_DAY = 50
MAX_TRADES_PER_HOUR = 10
MIN_TRADE_INTERVAL_SECONDS = 30
MAX_CONSECUTIVE_LOSSES = 5
MAX_DRAWDOWN_PERCENT = 15.0
EMERGENCY_STOP_CONDITIONS = {
    "consecutive_losses": MAX_CONSECUTIVE_LOSSES,
    "drawdown_percent": MAX_DRAWDOWN_PERCENT,
    "stale_model_hours": 168,  # 7 days
    "no_trades_warning_hours": 4
}

# MODEL VALIDATION SETTINGS
MIN_MODEL_CONFIDENCE = 0.6
ENSEMBLE_MODEL_REQUIRED = True  # Require ensemble models for production
VALIDATE_MODEL_ON_STARTUP = True

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Feature Engineering Settings
TECHNICAL_INDICATORS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ema_periods": [9, 21, 50],
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14
}

# Model Parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "min_samples_split": 8,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "oob_score": True,
    "bootstrap": True,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
    "eval_metric": "logloss"
}

LSTM_PARAMS = {
    "sequence_length": 20,
    "units": 50,
    "dropout": 0.2,
    "epochs": 50,
    "batch_size": 64,
    "validation_split": 0.2
}

def get_model_path(model_type: str, filename: str = "trained_model.pkl") -> Path:
    """Get the path for a specific model file."""
    model_dirs = {
        "rf": RF_MODEL_DIR,
        "random_forest": RF_MODEL_DIR,
        "xgb": XGB_MODEL_DIR,
        "xgboost": XGB_MODEL_DIR,
        "lstm": LSTM_MODEL_DIR
    }
    
    model_dir = model_dirs.get(model_type.lower(), MODELS_DIR)
    return model_dir / filename

def get_data_path(symbol: str = None) -> Path:
    """Get the path for data files."""
    if symbol:
        return PARQUET_DATA_DIR / symbol.lower() / f"{symbol.upper()}.parquet"
    return PARQUET_DATA_DIR

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = []
    missing_vars = []
    
    # Check required variables based on usage
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        missing_vars.extend(["BINANCE_API_KEY", "BINANCE_SECRET_KEY"])
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return True

# Configuration object for backwards compatibility
class Config:
    """Configuration object containing all settings"""
    
    # Trading settings
    PAPER_TRADING = PAPER_TRADING
    live_trading = live_trading
    DEFAULT_TRADE_AMOUNT = DEFAULT_TRADE_AMOUNT
    STOP_LOSS_PERCENT = STOP_LOSS_PERCENT
    TAKE_PROFIT_PERCENT = TAKE_PROFIT_PERCENT
    
    # API settings
    BINANCE_API_KEY = BINANCE_API_KEY
    BINANCE_SECRET_KEY = BINANCE_SECRET_KEY
    
    # Discord settings
    DISCORD_WEBHOOK = DISCORD_WEBHOOK
    DISCORD_BOT_TOKEN = DISCORD_BOT_TOKEN
    
    # File paths
    PROJECT_ROOT = PROJECT_ROOT
    DATA_ROOT = DATA_ROOT
    MODELS_DIR = MODELS_DIR
    PARQUET_DATA_DIR = PARQUET_DATA_DIR
    
    # Technical indicators
    TECHNICAL_INDICATORS = TECHNICAL_INDICATORS
    
    # Model parameters
    RANDOM_FOREST_PARAMS = RANDOM_FOREST_PARAMS
    XGBOOST_PARAMS = XGBOOST_PARAMS
    RANDOM_STATE = RANDOM_STATE
    
    # Safety parameters (defaults for production)
    max_position_size_percent = 10  # 10% of balance max
    min_free_disk_gb = 1.0  # Minimum 1GB free disk space
    max_trades_per_day = 20
    max_trades_per_hour = 5
    risk_percentage = 2.0

# Create config instance
config = Config()

if __name__ == "__main__":
    # Test configuration
    print("=== Money Printer Configuration ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print(f"Paper Trading: {PAPER_TRADING}")
    
    try:
        validate_environment()
        print("✅ Environment validation passed")
    except ValueError as e:
        print(f"❌ Environment validation failed: {e}")

#!/usr/bin/env python3
"""
Production Configuration Handler

Secure environment variable loading with validation and safety checks.
"""

import os
import logging
from decimal import Decimal
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Configuration-related errors"""
    pass

class SafeConfig:
    """Production-safe configuration handler"""
    
    def __init__(self, env_file: str = ".env"):
        """Initialize configuration with safety checks"""
        self.env_file = env_file
        self._load_environment()
        self._validate_config()
        
    def _load_environment(self):
        """Load environment variables safely"""
        try:
            # Load .env file if it exists
            if os.path.exists(self.env_file):
                load_dotenv(self.env_file)
                logger.info(f"✅ Loaded configuration from {self.env_file}")
            else:
                logger.warning(f"⚠️ Environment file {self.env_file} not found, using system environment")
                
        except Exception as e:
            raise ConfigError(f"Failed to load environment: {e}")
    
    def _validate_config(self):
        """Validate critical configuration values"""
        # Check required API keys
        if self.live_trading:
            if not self.binance_api_key or not self.binance_secret_key:
                raise ConfigError("Live trading enabled but missing Binance API credentials")
        
        # Validate numerical limits
        if self.max_daily_trades <= 0 or self.max_daily_trades > 1000:
            raise ConfigError(f"Invalid max_daily_trades: {self.max_daily_trades}")
            
        if not (0 <= self.bot_max_loss_percent <= 100):
            raise ConfigError(f"Invalid bot_max_loss_percent: {self.bot_max_loss_percent}")
            
        if not (0 <= self.reinvestment_percent <= 100):
            raise ConfigError(f"Invalid reinvestment_percent: {self.reinvestment_percent}")
    
    @property
    def binance_api_key(self) -> str:
        """Get Binance API key based on trading mode"""
        if self.live_trading:
            return os.getenv("BINANCE_API_KEY", "")
        else:
            return os.getenv("BINANCE_API_KEY_TESTNET", "")
    
    @property
    def binance_secret_key(self) -> str:
        """Get Binance secret key based on trading mode"""
        if self.live_trading:
            return os.getenv("BINANCE_SECRET_KEY", "")
        else:
            return os.getenv("BINANCE_SECRET_KEY_TESTNET", "")
    
    @property
    def live_trading(self) -> bool:
        """Check if live trading is enabled"""
        return os.getenv("LIVE_TRADING", "false").lower() in ("true", "1", "yes")
    
    @property
    def max_daily_trades(self) -> int:
        """Maximum trades per day"""
        return int(os.getenv("MAX_DAILY_TRADES", "50"))
    
    @property
    def max_hourly_trades(self) -> int:
        """Maximum trades per hour"""
        return int(os.getenv("MAX_HOURLY_TRADES", "10"))
    
    @property
    def bot_max_loss_percent(self) -> float:
        """Maximum loss percentage before bot shutdown"""
        return float(os.getenv("BOT_MAX_LOSS_PERCENT", "20.0"))
    
    @property
    def enable_profit_reinvestment(self) -> bool:
        """Enable automatic profit reinvestment"""
        return os.getenv("ENABLE_PROFIT_REINVESTMENT", "false").lower() in ("true", "1", "yes")
    
    @property
    def reinvestment_percent(self) -> float:
        """Percentage of profits to reinvest"""
        return float(os.getenv("REINVESTMENT_PERCENT", "50.0"))
    
    @property
    def min_model_winrate(self) -> float:
        """Minimum model win rate to continue trading"""
        return float(os.getenv("MIN_MODEL_WINRATE", "50.0"))
    
    @property
    def max_position_size_percent(self) -> float:
        """Maximum position size as percentage of balance"""
        return float(os.getenv("MAX_POSITION_SIZE_PERCENT", "5.0"))
    
    @property
    def websocket_timeout_minutes(self) -> int:
        """WebSocket timeout in minutes"""
        return int(os.getenv("WEBSOCKET_TIMEOUT_MINUTES", "10"))
    
    @property
    def api_retry_delay_seconds(self) -> int:
        """API retry delay in seconds"""
        return int(os.getenv("API_RETRY_DELAY_SECONDS", "30"))
    
    @property
    def auto_prune_logs_days(self) -> int:
        """Auto-prune logs older than X days"""
        return int(os.getenv("AUTO_PRUNE_LOGS_DAYS", "30"))
    
    @property
    def min_free_disk_gb(self) -> float:
        """Minimum free disk space in GB"""
        return float(os.getenv("MIN_FREE_DISK_GB", "5.0"))
    
    @property
    def discord_webhook_traders(self) -> str:
        """Discord webhook for trader notifications"""
        return os.getenv("DISCORD_WEBHOOK_TRADERS", "")
    
    @property
    def discord_webhook_trainers(self) -> str:
        """Discord webhook for trainer notifications"""
        return os.getenv("DISCORD_WEBHOOK_TRAINERS", "")
    
    @property
    def min_usdt_balance(self) -> float:
        """Minimum USDT balance required for trading"""
        return float(os.getenv("MIN_USDT_BALANCE", "3.0"))
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get all trading-related configuration"""
        return {
            "live_trading": self.live_trading,
            "max_daily_trades": self.max_daily_trades,
            "max_hourly_trades": self.max_hourly_trades,
            "bot_max_loss_percent": self.bot_max_loss_percent,
            "enable_profit_reinvestment": self.enable_profit_reinvestment,
            "reinvestment_percent": self.reinvestment_percent,
            "min_model_winrate": self.min_model_winrate,
            "max_position_size_percent": self.max_position_size_percent,
            "websocket_timeout_minutes": self.websocket_timeout_minutes,
            "api_retry_delay_seconds": self.api_retry_delay_seconds
        }
    
    def validate_runtime_safety(self) -> bool:
        """Validate runtime safety conditions"""
        try:
            # Check disk space
            import shutil
            _, _, free_bytes = shutil.disk_usage(".")
            free_gb = free_bytes / (1024**3)
            
            if free_gb < self.min_free_disk_gb:
                logger.error(f"⚠️ Low disk space: {free_gb:.2f}GB < {self.min_free_disk_gb}GB")
                return False
            
            # Check if trading is temporarily disabled
            disable_file = "TRADING_DISABLED.flag"
            if os.path.exists(disable_file):
                logger.warning(f"⚠️ Trading disabled by flag file: {disable_file}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safety validation failed: {e}")
            return False

# Global configuration instance
config = SafeConfig()

def get_config() -> SafeConfig:
    """Get the global configuration instance"""
    return config

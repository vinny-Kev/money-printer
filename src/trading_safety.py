#!/usr/bin/env python3
"""
Production Trading Safety Manager

Handles all safety checks, risk management, and fault tolerance for trading operations.
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import shutil

logger = logging.getLogger(__name__)

@dataclass
class TradeState:
    """Current state of a trading pair"""
    symbol: str
    is_active: bool = False
    last_trade_time: Optional[datetime] = None
    consecutive_losses: int = 0
    daily_trade_count: int = 0
    hourly_trade_count: int = 0
    total_pnl: float = 0.0
    locked_until: Optional[datetime] = None
    
class TradingSafetyManager:
    """Production-ready trading safety and risk management"""
    
    def __init__(self, config):
        self.config = config
        self.trade_states: Dict[str, TradeState] = {}
        self.active_trades_lock = threading.Lock()
        self.last_data_time = None
        self.websocket_failures = 0
        self.api_rate_limit_until = None
        self.bot_start_time = datetime.utcnow()
        self.daily_trade_count = 0
        self.hourly_trade_count = 0
        self.total_bot_pnl = 0.0
        self.daily_pnl = 0.0  # Daily PnL (resets each day)
        self.daily_winning_trades = 0
        self.daily_losing_trades = 0
        self.starting_balance = 1000.0  # Track starting balance for percentage calculations
        self.last_hour_reset = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        self.last_day_reset = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Load persistent state
        self._load_state()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _load_state(self):
        """Load persistent trading state"""
        try:
            state_file = "data/trading_state.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    
                # Restore counters
                self.daily_trade_count = data.get('daily_trade_count', 0)
                self.total_bot_pnl = data.get('total_bot_pnl', 0.0)
                self.daily_pnl = data.get('daily_pnl', 0.0)
                self.daily_winning_trades = data.get('daily_winning_trades', 0)
                self.daily_losing_trades = data.get('daily_losing_trades', 0)
                self.starting_balance = data.get('starting_balance', 1000.0)
                
                # Restore trade states
                for symbol, state_data in data.get('trade_states', {}).items():
                    self.trade_states[symbol] = TradeState(
                        symbol=symbol,
                        is_active=state_data.get('is_active', False),
                        last_trade_time=datetime.fromisoformat(state_data['last_trade_time']) if state_data.get('last_trade_time') else None,
                        consecutive_losses=state_data.get('consecutive_losses', 0),
                        daily_trade_count=state_data.get('daily_trade_count', 0),
                        total_pnl=state_data.get('total_pnl', 0.0),
                        locked_until=datetime.fromisoformat(state_data['locked_until']) if state_data.get('locked_until') else None
                    )
                    
                logger.info("✅ Trading state loaded from persistence")
        except Exception as e:
            logger.warning(f"Could not load trading state: {e}")
    
    def _save_state(self):
        """Save persistent trading state"""
        try:
            os.makedirs("data", exist_ok=True)
            state_data = {
                'daily_trade_count': self.daily_trade_count,
                'total_bot_pnl': self.total_bot_pnl,
                'daily_pnl': self.daily_pnl,
                'daily_winning_trades': self.daily_winning_trades,
                'daily_losing_trades': self.daily_losing_trades,
                'starting_balance': self.starting_balance,
                'last_save': datetime.utcnow().isoformat(),
                'trade_states': {}
            }
            
            for symbol, state in self.trade_states.items():
                state_data['trade_states'][symbol] = {
                    'is_active': state.is_active,
                    'last_trade_time': state.last_trade_time.isoformat() if state.last_trade_time else None,
                    'consecutive_losses': state.consecutive_losses,
                    'daily_trade_count': state.daily_trade_count,
                    'total_pnl': state.total_pnl,
                    'locked_until': state.locked_until.isoformat() if state.locked_until else None
                }
            
            with open("data/trading_state.json", 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save trading state: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def background_worker():
            while True:
                try:
                    self._reset_counters()
                    self._cleanup_locks()
                    self._save_state()
                    self._check_disk_space()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    logger.error(f"Background task error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=background_worker, daemon=True)
        thread.start()
    
    def _reset_counters(self):
        """Reset time-based counters"""
        now = datetime.utcnow()
        
        # Reset hourly counters
        if now >= self.last_hour_reset + timedelta(hours=1):
            self.hourly_trade_count = 0
            for state in self.trade_states.values():
                state.hourly_trade_count = 0
            self.last_hour_reset = now.replace(minute=0, second=0, microsecond=0)
            logger.info("🔄 Hourly trade counters reset")
        
        # Reset daily counters
        if now >= self.last_day_reset + timedelta(days=1):
            self.daily_trade_count = 0
            self.daily_pnl = 0.0
            self.daily_winning_trades = 0
            self.daily_losing_trades = 0
            for state in self.trade_states.values():
                state.daily_trade_count = 0
            self.last_day_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
            logger.info("🔄 Daily trade counters and PnL reset")
    
    def _cleanup_locks(self):
        """Remove expired symbol locks"""
        now = datetime.utcnow()
        for state in self.trade_states.values():
            if state.locked_until and now > state.locked_until:
                state.locked_until = None
                logger.info(f"🔓 Symbol {state.symbol} unlocked")
    
    def _check_disk_space(self):
        """Check available disk space"""
        try:
            _, _, free_bytes = shutil.disk_usage(".")
            free_gb = free_bytes / (1024**3)
            
            if free_gb < self.config.min_free_disk_gb:
                logger.warning(f"⚠️ Low disk space: {free_gb:.2f}GB")
                # Auto-prune old logs if enabled
                if self.config.auto_prune_logs_days > 0:
                    self._prune_old_files()
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
    
    def _prune_old_files(self):
        """Prune old log files and receipts"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.auto_prune_logs_days)
            pruned_count = 0
            
            # Prune old receipts
            receipts_dir = "src/trading_bot/receipts"
            if os.path.exists(receipts_dir):
                for filename in os.listdir(receipts_dir):
                    filepath = os.path.join(receipts_dir, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if file_time < cutoff_date:
                            os.remove(filepath)
                            pruned_count += 1
            
            if pruned_count > 0:
                logger.info(f"🗑️ Pruned {pruned_count} old files")
                
        except Exception as e:
            logger.error(f"File pruning failed: {e}")
    
    def calculate_dynamic_sl(self, predicted_profit: float, confidence_score: float) -> float:
        """Calculate dynamic stop loss based on prediction and confidence"""
        base_sl = 0.025  # 2.5% minimum
        confidence_factor = 1.0 - confidence_score  # lower confidence = higher SL (looser)
        predicted_margin_factor = max(0.5, predicted_profit / 0.08)  # scale based on potential reward

        sl_percent = base_sl + (confidence_factor * 0.05)  # up to +5%
        sl_percent *= predicted_margin_factor
        return round(sl_percent, 4)
    
    def calculate_position_size(self, balance: float, confidence: float, volatility: float) -> float:
        """Calculate safe position size with volatility adjustment for all balance levels"""
        # Base position size from config
        base_size_percent = self.config.max_position_size_percent / 100
        
        # Adjust for confidence (higher confidence = larger position)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_multiplier = max(0.3, 1.0 - (volatility * 2))  # 0.3x to 1.0x
        
        # Calculate final position size
        position_size = balance * base_size_percent * confidence_multiplier * volatility_multiplier
        
        # Special handling for small balances ($5-50)
        if balance <= 50:
            # For small balances, use a more aggressive position sizing to make trades possible
            min_position = max(3.0, balance * 0.2)  # Use at least 20% of balance, minimum $3
            # Cap at 80% of balance for very small accounts
            max_position = balance * 0.8
            position_size = max(min_position, min(position_size, max_position))
        else:
            # Ensure minimum position size for larger balances
            min_position = 10.0  # $10 minimum for balances > $50
            position_size = max(min_position, position_size)
        
        return position_size
    
    def can_afford_symbol(self, symbol: str, symbol_price: float, balance: float) -> Tuple[bool, str, Dict[str, float]]:
        """Check if user can afford to trade a symbol based on minimum order requirements"""
        try:
            # Get symbol information from exchange (this would need to be passed in or fetched)
            # For now, we'll use common minimum quantities for popular coins
            common_min_qtys = {
                # Major coins
                'BTCUSDT': 0.00001,   # ~$0.50 min
                'ETHUSDT': 0.0001,    # ~$0.30 min  
                'BNBUSDT': 0.001,     # ~$0.50 min
                'SOLUSDT': 0.001,     # ~$0.20 min
                'XRPUSDT': 1.0,       # ~$0.60 min
                
                # Top altcoins  
                'ADAUSDT': 1.0,       # ~$0.40 min
                'DOTUSDT': 0.1,       # ~$0.50 min
                'LINKUSDT': 0.01,     # ~$0.20 min
                'LTCUSDT': 0.001,     # ~$0.10 min
                'AVAXUSDT': 0.01,     # ~$0.40 min
                'ATOMUSDT': 0.01,     # ~$0.10 min
                'MATICUSDT': 1.0,     # ~$0.50 min
                'UNIUSDT': 0.01,      # ~$0.10 min
                'FILUSDT': 0.01,      # ~$0.50 min
                'VETUSDT': 10.0,      # ~$0.30 min
                
                # Popular mid-caps
                'SHIBUSDT': 400000.0, # ~$8.00 min (realistic high MOQ)
                'DOGEUSDT': 1.0,      # ~$0.40 min
                'TRXUSDT': 1.0,       # ~$0.20 min
                'NEARUSDT': 0.1,      # ~$0.50 min
                'FTMUSDT': 1.0,       # ~$0.80 min
                'SANDUSDT': 0.1,      # ~$0.30 min
                'MANAUSDT': 0.1,      # ~$0.50 min
                'CHZUSDT': 1.0,       # ~$0.10 min
                'ENJUSDT': 0.1,       # ~$0.20 min
                'GALAUSDT': 1.0,      # ~$0.04 min
                
                # Lower cap coins (often higher MOQ)
                'PEPEUSDT': 5000000.0,  # ~$5.00 min (realistic high MOQ meme coin)
                'FLOKIUSDT': 50000.0,   # ~$10.00 min (realistic high MOQ)
                'BONKUSDT': 200000.0,   # ~$6.00 min (realistic MOQ)
            }
            
            # Use conservative default minimum for unknown symbols
            # Default to 0.01 which works for most mid-cap coins
            min_qty = common_min_qtys.get(symbol, 0.01)
            min_order_value = min_qty * symbol_price
            
            # Add a buffer for fees and slippage (0.2% total)
            required_balance = min_order_value * 1.002
            
            # Calculate what percentage of balance this would use
            balance_percent = (required_balance / balance) * 100 if balance > 0 else 100
            
            # Prepare info for logging
            affordability_info = {
                'min_qty': min_qty,
                'min_order_value': min_order_value,
                'required_balance': required_balance,
                'balance_percent': balance_percent,
                'symbol_price': symbol_price
            }
            
            if required_balance > balance:
                return False, f"Minimum order (${min_order_value:.2f}) exceeds balance (${balance:.2f})", affordability_info
            
            # For very small balances, warn if single trade would use >50% of balance
            if balance <= 20 and balance_percent > 50:
                return False, f"Trade would use {balance_percent:.1f}% of small balance", affordability_info
            
            return True, f"Affordable - uses {balance_percent:.1f}% of balance", affordability_info
            
        except Exception as e:
            logger.error(f"Error checking affordability for {symbol}: {e}")
            return False, f"Error checking affordability: {e}", {}
    
    def check_volatility_filter(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, float]:
        """Check if symbol passes volatility filter using ATR"""
        try:
            # Calculate ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Calculate volatility as ATR percentage of close price
            volatility = (atr / df['close'].iloc[-1]) * 100
            
            # Filter out extremely volatile symbols (>15% ATR)
            max_volatility = 15.0
            
            if volatility > max_volatility:
                logger.info(f"⚠️ {symbol} filtered out - high volatility: {volatility:.2f}%")
                return False, volatility
            
            return True, volatility
            
        except Exception as e:
            logger.error(f"Volatility check failed for {symbol}: {e}")
            return False, 0.0
    
    def check_candle_body_confirmation(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check candle body confirmation to avoid massive wicks"""
        try:
            latest = df.iloc[-1]
            
            # Calculate candle metrics
            body_size = abs(latest['close'] - latest['open'])
            total_range = latest['high'] - latest['low']
            upper_wick = latest['high'] - max(latest['open'], latest['close'])
            lower_wick = min(latest['open'], latest['close']) - latest['low']
            
            if total_range == 0:
                return False
            
            # Calculate ratios
            body_ratio = body_size / total_range
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            
            # Filter criteria
            min_body_ratio = 0.4  # Body should be at least 40% of total range
            max_wick_ratio = 0.4  # No single wick should be >40% of range
            
            if body_ratio < min_body_ratio:
                logger.info(f"⚠️ {symbol} filtered out - small body ratio: {body_ratio:.2f}")
                return False
            
            if upper_wick_ratio > max_wick_ratio or lower_wick_ratio > max_wick_ratio:
                logger.info(f"⚠️ {symbol} filtered out - large wick ratio: {max(upper_wick_ratio, lower_wick_ratio):.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Candle confirmation failed for {symbol}: {e}")
            return False
    
    def can_trade_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Check if symbol can be traded based on all safety conditions"""
        with self.active_trades_lock:
            # Get or create trade state
            if symbol not in self.trade_states:
                self.trade_states[symbol] = TradeState(symbol=symbol)
            
            state = self.trade_states[symbol]
            now = datetime.utcnow()
            
            # Check if symbol is locked (cooldown period)
            if state.locked_until and now < state.locked_until:
                remaining = (state.locked_until - now).total_seconds() / 60
                return False, f"Symbol locked for {remaining:.1f} more minutes"
            
            # Check if already has active trade
            if state.is_active:
                return False, "Active trade already exists for this symbol"
            
            # Check daily limits
            if self.daily_trade_count >= self.config.max_daily_trades:
                return False, f"Daily trade limit reached ({self.config.max_daily_trades})"
            
            if state.daily_trade_count >= 5:  # Max 5 trades per symbol per day
                return False, "Daily symbol trade limit reached"
            
            # Check hourly limits
            if self.hourly_trade_count >= self.config.max_hourly_trades:
                return False, f"Hourly trade limit reached ({self.config.max_hourly_trades})"
            
            # Check consecutive losses
            if state.consecutive_losses >= 3:
                return False, "Too many consecutive losses for this symbol"
            
            # Check bot total loss
            loss_percent = (abs(self.total_bot_pnl) / 1000) * 100  # Assuming $1000 starting capital
            if self.total_bot_pnl < 0 and loss_percent > self.config.bot_max_loss_percent:
                return False, f"Bot max loss exceeded: {loss_percent:.1f}%"
            
            return True, "All checks passed"
    
    def can_trade_now(self) -> Tuple[bool, str]:
        """Check if bot can trade at all right now"""
        # Reset counters if needed
        self._reset_counters()

        # Check daily trade limits
        if self.daily_trade_count >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached ({self.config.max_daily_trades})"

        # Check hourly trade limits
        if self.hourly_trade_count >= self.config.max_hourly_trades:
            return False, f"Hourly trade limit reached ({self.config.max_hourly_trades})"

        # Check API rate limits
        if self.api_rate_limit_until and datetime.utcnow() < self.api_rate_limit_until:
            remaining = (self.api_rate_limit_until - datetime.utcnow()).total_seconds()
            return False, f"API rate limited for {remaining:.0f} more seconds"

        # Check WebSocket data freshness
        if self.last_data_time:
            data_age_minutes = (datetime.utcnow() - self.last_data_time).total_seconds() / 60
            if data_age_minutes > self.config.websocket_timeout_minutes:
                return False, f"Stale market data: {data_age_minutes:.1f} minutes old"

        # Check if trading is manually disabled
        if os.path.exists("TRADING_DISABLED.flag"):
            return False, "Trading manually disabled by flag file"

        # Check runtime safety conditions
        if not self.config.validate_runtime_safety():
            return False, "Runtime safety check failed"

        return True, "Ready to trade"
    
    def register_trade_start(self, symbol: str) -> bool:
        """Register the start of a new trade"""
        with self.active_trades_lock:
            if symbol not in self.trade_states:
                self.trade_states[symbol] = TradeState(symbol=symbol)
            
            state = self.trade_states[symbol]
            
            # Mark as active
            state.is_active = True
            state.last_trade_time = datetime.utcnow()
            
            # Increment counters
            self.daily_trade_count += 1
            self.hourly_trade_count += 1
            state.daily_trade_count += 1
            state.hourly_trade_count += 1
            
            logger.info(f"📈 Trade started: {symbol} (Daily: {self.daily_trade_count}/{self.config.max_daily_trades})")
            return True
    
    def register_trade_end(self, symbol: str, was_successful: bool, pnl: float):
        """Register the end of a trade"""
        with self.active_trades_lock:
            if symbol not in self.trade_states:
                return
            
            state = self.trade_states[symbol]
            
            # Mark as inactive
            state.is_active = False
            
            # Update PnL tracking
            state.total_pnl += pnl
            self.total_bot_pnl += pnl
            self.daily_pnl += pnl
            
            # Update win/loss counters
            if was_successful:
                self.daily_winning_trades += 1
            else:
                self.daily_losing_trades += 1
            
            # Update consecutive losses
            if was_successful:
                state.consecutive_losses = 0
            else:
                state.consecutive_losses += 1
                # Apply cooldown for losses
                cooldown_minutes = min(60, state.consecutive_losses * 10)  # 10-60 min cooldown
                state.locked_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
                logger.info(f"🔒 {symbol} locked for {cooldown_minutes} minutes after loss")
            
            logger.info(f"📊 Trade ended: {symbol} | Success: {was_successful} | P&L: {pnl:.2f} | Total: {self.total_bot_pnl:.2f}")
    
    def handle_api_rate_limit(self, retry_after_seconds: int = None):
        """Handle API rate limiting"""
        delay = retry_after_seconds or self.config.api_retry_delay_seconds
        self.api_rate_limit_until = datetime.utcnow() + timedelta(seconds=delay)
        logger.warning(f"🚫 API rate limited - backing off for {delay} seconds")
    
    def handle_websocket_failure(self):
        """Handle WebSocket connection failures"""
        self.websocket_failures += 1
        
        if self.websocket_failures >= 5:
            # Too many failures, pause trading
            pause_file = "TRADING_DISABLED.flag"
            with open(pause_file, 'w') as f:
                f.write(f"Auto-paused due to {self.websocket_failures} WebSocket failures at {datetime.utcnow().isoformat()}")
            
            logger.error(f"🚨 Trading auto-paused due to {self.websocket_failures} WebSocket failures")
            # Send alert notification here
    
    def update_data_timestamp(self):
        """Update the last data received timestamp"""
        self.last_data_time = datetime.utcnow()
        # Reset failure count on successful data
        self.websocket_failures = 0
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        active_trades = sum(1 for state in self.trade_states.values() if state.is_active)
        locked_symbols = sum(1 for state in self.trade_states.values() if state.locked_until and datetime.utcnow() < state.locked_until)
        
        # Calculate daily win rate
        daily_total_trades = self.daily_winning_trades + self.daily_losing_trades
        daily_win_rate = (self.daily_winning_trades / daily_total_trades * 100) if daily_total_trades > 0 else 0
        
        # Calculate daily PnL percentage
        daily_pnl_percent = (self.daily_pnl / self.starting_balance * 100) if self.starting_balance > 0 else 0
        
        # Calculate total PnL percentage
        total_pnl_percent = (self.total_bot_pnl / self.starting_balance * 100) if self.starting_balance > 0 else 0
        
        return {
            "bot_status": "ACTIVE" if self.can_trade_now()[0] else "PAUSED",
            "uptime_hours": (datetime.utcnow() - self.bot_start_time).total_seconds() / 3600,
            "total_pnl": self.total_bot_pnl,
            "total_pnl_percent": total_pnl_percent,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percent": daily_pnl_percent,
            "daily_win_rate": daily_win_rate,
            "daily_winning_trades": self.daily_winning_trades,
            "daily_losing_trades": self.daily_losing_trades,
            "starting_balance": self.starting_balance,
            "daily_trades": f"{self.daily_trade_count}/{self.config.max_daily_trades}",
            "hourly_trades": f"{self.hourly_trade_count}/{self.config.max_hourly_trades}",
            "active_trades": active_trades,
            "locked_symbols": locked_symbols,
            "websocket_failures": self.websocket_failures,
            "last_data_age_minutes": (datetime.utcnow() - self.last_data_time).total_seconds() / 60 if self.last_data_time else None,
            "api_rate_limited": bool(self.api_rate_limit_until and datetime.utcnow() < self.api_rate_limit_until)
        }

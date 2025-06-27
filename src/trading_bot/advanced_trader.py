"""
Advanced Trading Bot with Comprehensive Safety Measures and Modular Architecture
Implements all safety protocols, dynamic TP/SL, confidence-based position sizing, and real-time monitoring.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import pickle
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Local imports
from .trade_runner import get_client, get_current_price, place_order, clean_print
from ..safe_config import get_config
from ..discord_notifications import send_trader_notification


class TradingMode(Enum):
    """Trading mode enumeration"""
    DRY = "dry"
    TESTNET = "testnet"
    LIVE = "live"


@dataclass
class TradeSignal:
    """Trade signal with confidence and metadata"""
    symbol: str
    confidence: float
    predicted_profit: float
    predicted_loss: float
    entry_price: float
    volatility: float
    rsi: float
    macd: float
    volume_change: float
    timestamp: datetime
    model_name: str


@dataclass
class Trade:
    """Active trade representation"""
    id: str
    symbol: str
    quantity: float
    entry_price: float
    take_profit: float
    stop_loss: float
    confidence: float
    predicted_profit: float
    timestamp: datetime
    model_name: str
    status: str = "OPEN"
    current_price: float = 0.0
    current_pnl: float = 0.0
    current_pnl_percent: float = 0.0
    exit_price: float = 0.0
    exit_timestamp: Optional[datetime] = None
    trade_duration: Optional[timedelta] = None


@dataclass
class TradingStats:
    """Trading performance statistics"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    open_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    win_rate: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    average_trade: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    current_balance: float = 0.0
    starting_balance: float = 1000.0
    current_mode: str = "DRY"
    last_trade_time: Optional[datetime] = None


class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config):
        self.config = config
        self.max_concurrent_trades = getattr(config, 'max_concurrent_trades', 3)
        self.min_confidence = getattr(config, 'min_confidence_threshold', 0.6)
        self.max_consecutive_losses = getattr(config, 'max_consecutive_losses', 5)
        self.min_win_rate = getattr(config, 'min_win_rate', 0.5)
        self.max_daily_loss = getattr(config, 'max_daily_loss_percent', 0.1)  # 10%
        self.position_size_limits = getattr(config, 'position_size_limits', {'min': 0.001, 'max': 0.1})
        
        self.daily_loss = 0.0
        self.daily_loss_reset_time = datetime.now().date()
        
        self.logger = logging.getLogger("RiskManager")
    
    def check_trade_safety(self, signal: TradeSignal, stats: TradingStats, 
                          current_trades: List[Trade]) -> Tuple[bool, str]:
        """Comprehensive trade safety check"""
        
        # Check concurrent trades limit
        if len(current_trades) >= self.max_concurrent_trades:
            return False, f"Max concurrent trades reached ({self.max_concurrent_trades})"
        
        # Check confidence threshold
        if signal.confidence < self.min_confidence:
            return False, f"Confidence too low ({signal.confidence:.1%} < {self.min_confidence:.1%})"
        
        # Check consecutive losses
        if stats.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Too many consecutive losses ({stats.consecutive_losses})"
        
        # Check win rate (only after sufficient trades)
        if stats.total_trades >= 10 and stats.win_rate < self.min_win_rate:
            return False, f"Win rate too low ({stats.win_rate:.1%} < {self.min_win_rate:.1%})"
        
        # Check daily loss limit
        self._update_daily_loss(stats)
        if self.daily_loss >= self.max_daily_loss:
            return False, f"Daily loss limit reached ({self.daily_loss:.1%})"
        
        # Check if symbol already being traded
        if any(trade.symbol == signal.symbol for trade in current_trades):
            return False, f"Already trading {signal.symbol}"
        
        return True, "All safety checks passed"
    
    def calculate_position_size(self, balance: float, confidence: float, 
                              volatility: float) -> float:
        """Calculate position size based on confidence and risk"""
        
        # Base position size (percentage of balance)
        base_size = 0.02  # 2% of balance
        
        # Confidence multiplier (0.5x to 2x based on confidence)
        confidence_multiplier = 0.5 + (confidence - 0.5) * 3
        confidence_multiplier = max(0.5, min(2.0, confidence_multiplier))
        
        # Volatility adjustment (reduce size for high volatility)
        volatility_multiplier = max(0.3, 1.0 - (volatility * 2))
        
        # Calculate final position size
        position_percent = base_size * confidence_multiplier * volatility_multiplier
        
        # Apply limits
        position_percent = max(self.position_size_limits['min'], 
                             min(self.position_size_limits['max'], position_percent))
        
        position_size = balance * position_percent
        
        self.logger.info(f"Position sizing: Base={base_size:.1%}, "
                        f"Confidence={confidence_multiplier:.2f}, "
                        f"Volatility={volatility_multiplier:.2f}, "
                        f"Final={position_percent:.1%} (${position_size:.2f})")
        
        return position_size
    
    def calculate_dynamic_tp_sl(self, signal: TradeSignal) -> Tuple[float, float]:
        """Calculate dynamic TP/SL based on signal strength and volatility"""
        
        # Base TP/SL percentages
        base_tp = 0.03  # 3%
        base_sl = 0.02  # 2%
        
        # Adjust based on confidence
        confidence_factor = signal.confidence
        tp_multiplier = 0.5 + confidence_factor * 1.5  # 0.5x to 2x
        sl_multiplier = 2.0 - confidence_factor  # 2x to 1x (higher confidence = tighter SL)
        
        # Adjust based on volatility
        volatility_factor = signal.volatility / 100  # Convert percentage to decimal
        tp_volatility_adj = 1.0 + volatility_factor  # Higher volatility = wider TP
        sl_volatility_adj = 1.0 + volatility_factor * 0.5  # Higher volatility = wider SL
        
        # Calculate final TP/SL
        take_profit_percent = base_tp * tp_multiplier * tp_volatility_adj
        stop_loss_percent = base_sl * sl_multiplier * sl_volatility_adj
        
        # Apply reasonable limits
        take_profit_percent = max(0.01, min(0.15, take_profit_percent))  # 1% to 15%
        stop_loss_percent = max(0.005, min(0.1, stop_loss_percent))  # 0.5% to 10%
        
        self.logger.info(f"Dynamic TP/SL for {signal.symbol}: "
                        f"TP={take_profit_percent:.1%}, SL={stop_loss_percent:.1%}")
        
        return take_profit_percent, stop_loss_percent
    
    def _update_daily_loss(self, stats: TradingStats):
        """Update daily loss tracking"""
        current_date = datetime.now().date()
        
        # Reset daily loss if new day
        if current_date != self.daily_loss_reset_time:
            self.daily_loss = 0.0
            self.daily_loss_reset_time = current_date
        
        # Calculate today's loss percentage
        if stats.total_pnl < 0:
            self.daily_loss = abs(stats.total_pnl_percent)


class PnLTracker:
    """PnL and performance tracking system"""
    
    def __init__(self, starting_balance: float = 1000.0):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.trades_history: List[Dict] = []
        self.csv_path = "trading_performance.csv"
        
        self.logger = logging.getLogger("PnLTracker")
        
        # Initialize CSV if not exists
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file for trade logging"""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'trade_id', 'timestamp', 'symbol', 'action', 'quantity', 
                'entry_price', 'exit_price', 'take_profit', 'stop_loss',
                'confidence', 'predicted_profit', 'actual_pnl', 'actual_pnl_percent',
                'trade_duration_minutes', 'model_name', 'mode', 'success'
            ])
            df.to_csv(self.csv_path, index=False)
            self.logger.info(f"Initialized trading CSV: {self.csv_path}")
    
    def log_trade(self, trade: Trade, mode: str):
        """Log completed trade to CSV and memory"""
        
        trade_data = {
            'trade_id': trade.id,
            'timestamp': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'action': 'BUY_SELL',
            'quantity': trade.quantity,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'take_profit': trade.take_profit,
            'stop_loss': trade.stop_loss,
            'confidence': trade.confidence,
            'predicted_profit': trade.predicted_profit,
            'actual_pnl': trade.current_pnl,
            'actual_pnl_percent': trade.current_pnl_percent,
            'trade_duration_minutes': trade.trade_duration.total_seconds() / 60 if trade.trade_duration else 0,
            'model_name': trade.model_name,
            'mode': mode,
            'success': trade.current_pnl > 0
        }
        
        # Save to memory
        self.trades_history.append(trade_data)
        
        # Append to CSV
        df = pd.DataFrame([trade_data])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        
        # Update balance
        self.current_balance += trade.current_pnl
        
        self.logger.info(f"Logged trade {trade.id}: {trade.symbol} PnL=${trade.current_pnl:.2f}")
    
    def get_stats(self, mode: str) -> TradingStats:
        """Calculate comprehensive trading statistics"""
        
        if not self.trades_history:
            return TradingStats(
                current_balance=self.current_balance,
                starting_balance=self.starting_balance,
                current_mode=mode
            )
        
        # Filter trades by mode if needed
        trades = self.trades_history
        
        total_trades = len(trades)
        successful_trades = sum(1 for t in trades if t['success'])
        failed_trades = total_trades - successful_trades
        
        total_pnl = sum(t['actual_pnl'] for t in trades)
        total_pnl_percent = (total_pnl / self.starting_balance) * 100
        
        win_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        pnl_values = [t['actual_pnl'] for t in trades]
        best_trade = max(pnl_values) if pnl_values else 0
        worst_trade = min(pnl_values) if pnl_values else 0
        average_trade = np.mean(pnl_values) if pnl_values else 0
        
        # Calculate consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for trade in reversed(trades):
            if not trade['success']:
                current_streak += 1
                consecutive_losses = current_streak
            else:
                if current_streak > max_consecutive_losses:
                    max_consecutive_losses = current_streak
                current_streak = 0
        
        last_trade_time = None
        if trades:
            last_trade_time = datetime.fromisoformat(trades[-1]['timestamp'])
        
        return TradingStats(
            total_trades=total_trades,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            win_rate=win_rate,
            best_trade=best_trade,
            worst_trade=worst_trade,
            average_trade=average_trade,
            consecutive_losses=consecutive_losses,
            max_consecutive_losses=max_consecutive_losses,
            current_balance=self.current_balance,
            starting_balance=self.starting_balance,
            current_mode=mode,
            last_trade_time=last_trade_time
        )


class Strategy:
    """Trading strategy with ML predictions and signal generation"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.6):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.expected_features = []
        
        self.logger = logging.getLogger("Strategy")
        
        self._load_model()
        self._load_expected_features()
    
    def _load_model(self):
        """Load trained ML model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            self.model = None
    
    def _load_expected_features(self):
        """Load expected features for the model"""
        try:
            features_path = os.path.join(os.path.dirname(self.model_path), "expected_features.json")
            with open(features_path, 'r') as f:
                self.expected_features = json.load(f)
            self.logger.info(f"✅ Loaded {len(self.expected_features)} expected features")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load expected features: {e}")
            self.expected_features = []
    
    def generate_signals(self, market_data: pd.DataFrame) -> List[TradeSignal]:
        """Generate trade signals from market data"""
        
        if self.model is None:
            self.logger.error("❌ No model available for signal generation")
            return []
        
        signals = []
        
        try:
            # Prepare features for prediction
            features_df = self._prepare_features(market_data)
            
            if features_df.empty:
                return signals
            
            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_df)
                confidences = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            else:
                predictions = self.model.predict(features_df)
                confidences = predictions
            
            # Generate signals for each symbol
            for idx, (_, row) in enumerate(market_data.iterrows()):
                confidence = float(confidences[idx])
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Calculate predicted profit/loss based on confidence and technical indicators
                predicted_profit = self._calculate_predicted_profit(row, confidence)
                predicted_loss = self._calculate_predicted_loss(row, confidence)
                
                signal = TradeSignal(
                    symbol=row['symbol'],
                    confidence=confidence,
                    predicted_profit=predicted_profit,
                    predicted_loss=predicted_loss,
                    entry_price=row['close'],
                    volatility=row.get('volatility', 0),
                    rsi=row.get('rsi', 50),
                    macd=row.get('macd', 0),
                    volume_change=row.get('volume_change', 0),
                    timestamp=datetime.now(),
                    model_name="random_forest_v1"
                )
                
                signals.append(signal)
            
            # Sort signals by confidence (best first)
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            self.logger.info(f"Generated {len(signals)} signals from {len(market_data)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def _prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction"""
        
        if not self.expected_features:
            # Use all numeric columns if no expected features
            numeric_cols = market_data.select_dtypes(include=[np.number]).columns
            return market_data[numeric_cols].fillna(0)
        
        # Check for missing features
        missing_features = [f for f in self.expected_features if f not in market_data.columns]
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            # Create dummy features with zeros
            for feature in missing_features:
                market_data[feature] = 0
        
        # Select only expected features
        feature_data = market_data[self.expected_features].fillna(0)
        
        # Remove infinite values
        feature_data = feature_data.replace([np.inf, -np.inf], 0)
        
        return feature_data
    
    def _calculate_predicted_profit(self, row: pd.Series, confidence: float) -> float:
        """Calculate predicted profit percentage"""
        base_profit = 0.02  # 2% base
        confidence_bonus = (confidence - 0.5) * 0.08  # Up to 4% bonus
        
        # Technical indicator adjustments
        rsi = row.get('rsi', 50)
        if rsi < 30:  # Oversold
            confidence_bonus += 0.01
        elif rsi > 70:  # Overbought
            confidence_bonus -= 0.01
        
        return max(0.005, base_profit + confidence_bonus)  # Minimum 0.5%
    
    def _calculate_predicted_loss(self, row: pd.Series, confidence: float) -> float:
        """Calculate predicted loss percentage"""
        base_loss = 0.015  # 1.5% base
        confidence_adjustment = (1 - confidence) * 0.02  # Higher loss for lower confidence
        
        return base_loss + confidence_adjustment


class TradeExecutor:
    """Execute and monitor trades across different modes"""
    
    def __init__(self, mode: TradingMode, config):
        self.mode = mode
        self.config = config
        self.client = None
        self.dry_balance = getattr(config, 'dry_trade_budget', 1000.0)
        
        self.logger = logging.getLogger("TradeExecutor")
        
        if mode != TradingMode.DRY:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Binance client based on mode"""
        try:
            if self.mode == TradingMode.TESTNET:
                self.client = get_client()
                # Set testnet URL
                self.client.API_URL = "https://testnet.binance.vision/api"
                self.logger.info("✅ Testnet client initialized")
            elif self.mode == TradingMode.LIVE:
                self.client = get_client()
                self.logger.info("✅ Live trading client initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize client: {e}")
            self.client = None
    
    def get_balance(self) -> float:
        """Get current USDT balance"""
        if self.mode == TradingMode.DRY:
            return self.dry_balance
        
        if not self.client:
            return 0.0
        
        try:
            balance_info = self.client.get_asset_balance(asset="USDT")
            return float(balance_info["free"]) if balance_info else 0.0
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def execute_trade(self, signal: TradeSignal, position_size: float, 
                     take_profit_percent: float, stop_loss_percent: float) -> Optional[Trade]:
        """Execute a trade based on signal"""
        
        try:
            # Calculate trade parameters
            quantity = position_size / signal.entry_price
            take_profit_price = signal.entry_price * (1 + take_profit_percent)
            stop_loss_price = signal.entry_price * (1 - stop_loss_percent)
            
            # Generate unique trade ID
            trade_id = f"{signal.symbol}_{int(signal.timestamp.timestamp())}"
            
            # Execute buy order
            if self.mode == TradingMode.DRY:
                # Dry trading simulation
                if self.dry_balance < position_size:
                    self.logger.warning(f"Insufficient dry balance: ${self.dry_balance:.2f} < ${position_size:.2f}")
                    return None
                
                self.dry_balance -= position_size
                clean_print(f"[DRY] Bought {quantity:.8f} {signal.symbol} @ ${signal.entry_price:.6f}", "SUCCESS")
                
            else:
                # Real trading (testnet or live)
                order_result = place_order("BUY", signal.symbol, quantity, signal.entry_price)
                if not order_result.get("success", False):
                    self.logger.error(f"Buy order failed: {order_result.get('error', 'Unknown error')}")
                    return None
            
            # Create trade object
            trade = Trade(
                id=trade_id,
                symbol=signal.symbol,
                quantity=quantity,
                entry_price=signal.entry_price,
                take_profit=take_profit_price,
                stop_loss=stop_loss_price,
                confidence=signal.confidence,
                predicted_profit=signal.predicted_profit,
                timestamp=signal.timestamp,
                model_name=signal.model_name,
                current_price=signal.entry_price
            )
            
            self.logger.info(f"✅ Trade executed: {trade.symbol} ${position_size:.2f}")
            clean_print(f"Trade opened: {trade.symbol} - ${position_size:.2f} @ {signal.confidence:.1%} confidence", "SUCCESS")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def update_trade(self, trade: Trade) -> Trade:
        """Update trade with current market price and PnL"""
        try:
            current_price = get_current_price(trade.symbol)
            if current_price is None:
                return trade
            
            trade.current_price = current_price
            
            # Calculate current PnL
            if self.mode == TradingMode.DRY:
                current_value = trade.quantity * current_price
                entry_value = trade.quantity * trade.entry_price
                trade.current_pnl = current_value - entry_value
            else:
                # For real trading, account for fees
                current_value = trade.quantity * current_price * 0.999  # Subtract fee
                entry_value = trade.quantity * trade.entry_price * 1.001  # Add fee
                trade.current_pnl = current_value - entry_value
            
            trade.current_pnl_percent = (trade.current_pnl / (trade.quantity * trade.entry_price)) * 100
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error updating trade {trade.id}: {e}")
            return trade
    
    def close_trade(self, trade: Trade, reason: str = "TP/SL") -> Trade:
        """Close a trade and update final PnL"""
        try:
            trade.exit_price = trade.current_price
            trade.exit_timestamp = datetime.now()
            trade.trade_duration = trade.exit_timestamp - trade.timestamp
            trade.status = "CLOSED"
            
            # Execute sell order
            if self.mode == TradingMode.DRY:
                self.dry_balance += trade.quantity * trade.current_price
                clean_print(f"[DRY] Sold {trade.quantity:.8f} {trade.symbol} @ ${trade.current_price:.6f}", "SUCCESS")
            else:
                order_result = place_order("SELL", trade.symbol, trade.quantity, trade.current_price)
                if not order_result.get("success", False):
                    self.logger.warning(f"Sell order failed for {trade.id}, using current price for PnL calculation")
            
            duration_str = f"{trade.trade_duration.total_seconds()/60:.1f}m"
            pnl_status = "✅" if trade.current_pnl > 0 else "❌"
            
            clean_print(f"Trade closed: {trade.symbol} {pnl_status} PnL: ${trade.current_pnl:.2f} "
                       f"({trade.current_pnl_percent:+.1f}%) - {duration_str} - {reason}", 
                       "SUCCESS" if trade.current_pnl > 0 else "ERROR")
            
            self.logger.info(f"Trade closed: {trade.id} PnL=${trade.current_pnl:.2f} Reason={reason}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error closing trade {trade.id}: {e}")
            return trade


class AdvancedTrader:
    """Main trading bot orchestrator with comprehensive safety and monitoring"""
    
    def __init__(self, mode: TradingMode = TradingMode.DRY, config=None):
        self.mode = mode
        self.config = config or get_config()
        
        # Initialize components
        self.risk_manager = RiskManager(self.config)
        self.pnl_tracker = PnLTracker(getattr(self.config, 'starting_balance', 1000.0))
        self.strategy = Strategy(
            model_path=getattr(self.config, 'model_path', 'trained_model.pkl'),
            confidence_threshold=getattr(self.config, 'confidence_threshold', 0.6)
        )
        self.executor = TradeExecutor(mode, self.config)
        
        # Trading state
        self.active_trades: List[Trade] = []
        self.is_running = False
        self.is_paused = False
        self.pause_reason = ""
        
        # Monitoring
        self.monitor_thread = None
        self.monitor_interval = getattr(self.config, 'monitor_interval', 30)  # 30 seconds
        
        self.logger = logging.getLogger("AdvancedTrader")
        self.logger.info(f"🚀 Advanced Trader initialized in {mode.value.upper()} mode")
    
    def start_trading(self):
        """Start the trading bot with monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.is_paused = False
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_trades, daemon=True)
        self.monitor_thread.start()
        
        clean_print(f"🚀 Advanced Trading Bot Started in {self.mode.value.upper()} mode", "SUCCESS")
        self.logger.info("Trading bot started with real-time monitoring")
        
        # Send notification
        send_trader_notification(f"🚀 **Trading Bot Started** - Mode: {self.mode.value.upper()}")
    
    def stop_trading(self):
        """Stop the trading bot"""
        self.is_running = False
        
        # Close all open trades
        for trade in self.active_trades[:]:
            self.close_trade(trade, "MANUAL_STOP")
        
        clean_print("🛑 Trading Bot Stopped", "WARNING")
        self.logger.info("Trading bot stopped")
        
        send_trader_notification("🛑 **Trading Bot Stopped** - All positions closed")
    
    def pause_trading(self, reason: str):
        """Pause trading due to risk conditions"""
        self.is_paused = True
        self.pause_reason = reason
        
        clean_print(f"⏸️ Trading Paused: {reason}", "WARNING")
        self.logger.warning(f"Trading paused: {reason}")
        
        send_trader_notification(f"⏸️ **Trading Paused**: {reason}")
    
    def resume_trading(self):
        """Resume trading"""
        self.is_paused = False
        self.pause_reason = ""
        
        clean_print("▶️ Trading Resumed", "SUCCESS")
        self.logger.info("Trading resumed")
        
        send_trader_notification("▶️ **Trading Resumed**")
    
    def execute_single_trade(self, market_data: pd.DataFrame) -> Optional[Trade]:
        """Execute a single trade with full safety checks"""
        if not self.is_running or self.is_paused:
            return None
        
        try:
            # Generate signals
            signals = self.strategy.generate_signals(market_data)
            if not signals:
                self.logger.info("No valid signals generated")
                return None
            
            # Get best signal
            best_signal = signals[0]
            
            # Get current stats
            stats = self.get_status()
            
            # Safety checks
            is_safe, safety_reason = self.risk_manager.check_trade_safety(
                best_signal, stats, self.active_trades
            )
            
            if not is_safe:
                self.logger.info(f"Trade blocked by safety: {safety_reason}")
                
                # Check if we should pause trading
                if "consecutive losses" in safety_reason.lower() or "win rate" in safety_reason.lower():
                    self.pause_trading(safety_reason)
                
                return None
            
            # Calculate position size
            balance = self.executor.get_balance()
            position_size = self.risk_manager.calculate_position_size(
                balance, best_signal.confidence, best_signal.volatility
            )
            
            # Calculate dynamic TP/SL
            tp_percent, sl_percent = self.risk_manager.calculate_dynamic_tp_sl(best_signal)
            
            # Execute trade
            trade = self.executor.execute_trade(
                best_signal, position_size, tp_percent, sl_percent
            )
            
            if trade:
                self.active_trades.append(trade)
                
                # Send notification
                send_trader_notification(
                    f"💰 **New Trade Opened**\n"
                    f"Symbol: {trade.symbol}\n"
                    f"Size: ${position_size:.2f}\n"
                    f"Confidence: {trade.confidence:.1%}\n"
                    f"TP: ${trade.take_profit:.4f} | SL: ${trade.stop_loss:.4f}"
                )
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing single trade: {e}")
            return None
    
    def close_trade(self, trade: Trade, reason: str = "MANUAL") -> Trade:
        """Close a specific trade"""
        try:
            # Update trade with current price
            updated_trade = self.executor.update_trade(trade)
            
            # Close the trade
            closed_trade = self.executor.close_trade(updated_trade, reason)
            
            # Remove from active trades
            if trade in self.active_trades:
                self.active_trades.remove(trade)
            
            # Log to PnL tracker
            self.pnl_tracker.log_trade(closed_trade, self.mode.value)
            
            return closed_trade
            
        except Exception as e:
            self.logger.error(f"Error closing trade {trade.id}: {e}")
            return trade
    
    def _monitor_trades(self):
        """Background thread to monitor all active trades"""
        while self.is_running:
            try:
                if not self.is_paused and self.active_trades:
                    trades_to_close = []
                    
                    for trade in self.active_trades:
                        # Update trade with current price
                        updated_trade = self.executor.update_trade(trade)
                        
                        # Check TP/SL conditions
                        if updated_trade.current_price >= updated_trade.take_profit:
                            trades_to_close.append((updated_trade, "TAKE_PROFIT"))
                        elif updated_trade.current_price <= updated_trade.stop_loss:
                            trades_to_close.append((updated_trade, "STOP_LOSS"))
                        
                        # Check for signal reversal (optional advanced feature)
                        # TODO: Implement signal reversal detection
                    
                    # Close trades that hit TP/SL
                    for trade, reason in trades_to_close:
                        self.close_trade(trade, reason)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in trade monitoring: {e}")
                time.sleep(self.monitor_interval)
    
    def get_status(self) -> TradingStats:
        """Get comprehensive trading status"""
        stats = self.pnl_tracker.get_stats(self.mode.value)
        stats.open_trades = len(self.active_trades)
        stats.current_balance = self.executor.get_balance()
        
        return stats
    
    def get_discord_status(self) -> str:
        """Get Discord-formatted status message"""
        stats = self.get_status()
        
        # Status emojis
        mode_emoji = {"dry": "🧪", "testnet": "🌐", "live": "🚀"}
        status_emoji = "⏸️" if self.is_paused else ("🟢" if self.is_running else "🔴")
        
        # PnL color indicator
        pnl_emoji = "📈" if stats.total_pnl >= 0 else "📉"
        
        # Win rate color
        win_rate_emoji = "🎯" if stats.win_rate >= 0.6 else ("⚡" if stats.win_rate >= 0.4 else "⚠️")
        
        status_text = f"""
**{mode_emoji.get(self.mode.value, '🤖')} ADVANCED TRADING BOT STATUS {status_emoji}**

**💰 Balance & Performance**
Current Balance: `${stats.current_balance:.2f}`
Starting Balance: `${stats.starting_balance:.2f}`
{pnl_emoji} Total PnL: `${stats.total_pnl:+.2f}` (`{stats.total_pnl_percent:+.1f}%`)

**📊 Trading Statistics**
{win_rate_emoji} Win Rate: `{stats.win_rate:.1%}` ({stats.successful_trades}/{stats.total_trades})
📈 Best Trade: `${stats.best_trade:.2f}`
📉 Worst Trade: `${stats.worst_trade:.2f}`
📊 Average Trade: `${stats.average_trade:.2f}`

**🎯 Current Activity**
Active Trades: `{stats.open_trades}`
Consecutive Losses: `{stats.consecutive_losses}`
Mode: `{stats.current_mode.upper()}`
Status: `{"PAUSED" if self.is_paused else ("RUNNING" if self.is_running else "STOPPED")}`

**📅 Recent Activity**
Last Trade: `{stats.last_trade_time.strftime('%Y-%m-%d %H:%M') if stats.last_trade_time else 'None'}`
"""
        
        if self.is_paused:
            status_text += f"\n⚠️ **Paused Reason:** {self.pause_reason}"
        
        if self.active_trades:
            status_text += "\n\n**📊 Open Positions:**"
            for trade in self.active_trades[:3]:  # Show max 3 trades
                updated_trade = self.executor.update_trade(trade)
                duration = (datetime.now() - trade.timestamp).total_seconds() / 60
                status_text += f"\n• {trade.symbol}: {updated_trade.current_pnl_percent:+.1f}% ({duration:.0f}m)"
        
        return status_text
    
    def force_close_all_trades(self):
        """Emergency function to close all trades"""
        for trade in self.active_trades[:]:
            self.close_trade(trade, "EMERGENCY_CLOSE")
        
        clean_print("🚨 All trades force-closed", "WARNING")
        send_trader_notification("🚨 **Emergency**: All trades force-closed")


# Factory function for easy instantiation
def create_advanced_trader(mode: str = "dry", config=None) -> AdvancedTrader:
    """Create an AdvancedTrader instance with the specified mode"""
    
    trading_mode = TradingMode(mode.lower())
    return AdvancedTrader(trading_mode, config)


# Example usage functions for Discord integration
def start_advanced_trading(mode: str = "dry") -> AdvancedTrader:
    """Start advanced trading bot"""
    trader = create_advanced_trader(mode)
    trader.start_trading()
    return trader


def get_trading_status(trader: AdvancedTrader) -> str:
    """Get Discord-formatted trading status"""
    return trader.get_discord_status()


def execute_smart_trade(trader: AdvancedTrader, market_data: pd.DataFrame) -> Optional[Trade]:
    """Execute a single smart trade with full safety"""
    return trader.execute_single_trade(market_data)

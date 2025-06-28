"""
Base Trader Class
Abstract base class for all trading strategies with common functionality
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseTrader(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, 
                 binance_client=None,
                 storage_manager=None,
                 symbol: str = 'BTCUSDT',
                 trading_amount: float = 10.0,
                 risk_percentage: float = 2.0,
                 stop_loss_percentage: float = 2.0,
                 take_profit_percentage: float = 4.0):
        """
        Initialize base trader
        
        Args:
            binance_client: Enhanced Binance client instance
            storage_manager: Storage manager for data access
            symbol: Trading pair symbol
            trading_amount: Amount to trade in USDT
            risk_percentage: Maximum risk per trade (%)
            stop_loss_percentage: Stop loss threshold (%)
            take_profit_percentage: Take profit threshold (%)
        """
        self.binance_client = binance_client
        self.storage_manager = storage_manager
        self.symbol = symbol
        self.trading_amount = trading_amount
        self.risk_percentage = risk_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        
        # Trading state
        self.position = None  # None, 'long', 'short'
        self.entry_price = None
        self.entry_time = None
        self.stop_loss_price = None
        self.take_profit_price = None
        
        # Performance tracking
        self.trades_history = []
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info(f"🤖 {self.__class__.__name__} initialized for {symbol}")
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load trained model from file"""
        pass
    
    @abstractmethod
    def predict_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal from market data
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dict with keys: signal ('buy', 'sell', 'hold'), confidence, price_target
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict[str, Any], balance: float) -> float:
        """Calculate position size based on signal and risk management"""
        pass
    
    def get_market_data(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get recent market data for analysis"""
        try:
            if self.binance_client:
                # Get klines from Binance
                klines = self.binance_client.get_klines(
                    symbol=self.symbol,
                    interval='1m',
                    limit=limit
                )
                
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert to numeric and datetime
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    return df
            
            # Fallback to storage if available
            if self.storage_manager:
                logger.info("📁 Falling back to stored data")
                return self.storage_manager.get_latest_data(symbol=self.symbol, limit=limit)
                
        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
        
        return None
    
    def execute_trade(self, signal: Dict[str, Any]) -> bool:
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal from predict_signal()
            
        Returns:
            bool: True if trade executed successfully
        """
        try:
            if not self.binance_client:
                logger.warning("⚠️ Paper trading mode - no actual trades executed")
                return self._execute_paper_trade(signal)
            
            # Get current balance
            balance = self.binance_client.get_usdt_balance()
            if not balance:
                logger.error("❌ Could not get balance")
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(signal, balance)
            if position_size <= 0:
                logger.warning("⚠️ Position size too small, skipping trade")
                return False
            
            # Execute based on signal
            if signal['signal'] == 'buy' and self.position is None:
                return self._execute_buy_order(position_size, signal)
            elif signal['signal'] == 'sell' and self.position == 'long':
                return self._execute_sell_order(signal)
                
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
        
        return False
    
    def _execute_buy_order(self, position_size: float, signal: Dict[str, Any]) -> bool:
        """Execute buy order"""
        try:
            # Place market buy order
            order = self.binance_client.place_market_order(
                symbol=self.symbol,
                side='BUY',
                quantity=position_size
            )
            
            if order and order.get('status') == 'FILLED':
                self.position = 'long'
                self.entry_price = float(order['fills'][0]['price'])
                self.entry_time = datetime.now()
                
                # Set stop loss and take profit
                self.stop_loss_price = self.entry_price * (1 - self.stop_loss_percentage / 100)
                self.take_profit_price = self.entry_price * (1 + self.take_profit_percentage / 100)
                
                logger.info(f"✅ BUY order executed: {position_size} {self.symbol} at {self.entry_price}")
                logger.info(f"🛑 Stop Loss: {self.stop_loss_price:.4f}")
                logger.info(f"🎯 Take Profit: {self.take_profit_price:.4f}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Buy order failed: {e}")
        
        return False
    
    def _execute_sell_order(self, signal: Dict[str, Any]) -> bool:
        """Execute sell order"""
        try:
            # Get current position size
            balance = self.binance_client.get_asset_balance(self.symbol.replace('USDT', ''))
            if not balance or balance['free'] <= 0:
                logger.error("❌ No position to sell")
                return False
            
            # Place market sell order
            order = self.binance_client.place_market_order(
                symbol=self.symbol,
                side='SELL',
                quantity=balance['free']
            )
            
            if order and order.get('status') == 'FILLED':
                exit_price = float(order['fills'][0]['price'])
                pnl = (exit_price - self.entry_price) / self.entry_price * 100
                
                # Record trade
                trade_record = {
                    'entry_time': self.entry_time,
                    'exit_time': datetime.now(),
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'pnl_percentage': pnl,
                    'signal_confidence': signal.get('confidence', 0.0)
                }
                
                self.trades_history.append(trade_record)
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                
                # Reset position
                self.position = None
                self.entry_price = None
                self.entry_time = None
                self.stop_loss_price = None
                self.take_profit_price = None
                
                logger.info(f"✅ SELL order executed at {exit_price} (PnL: {pnl:.2f}%)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Sell order failed: {e}")
        
        return False
    
    def _execute_paper_trade(self, signal: Dict[str, Any]) -> bool:
        """Execute paper trade for testing"""
        # Get current price
        data = self.get_market_data(limit=1)
        if data is None or data.empty:
            return False
        
        current_price = data.iloc[-1]['close']
        
        if signal['signal'] == 'buy' and self.position is None:
            self.position = 'long'
            self.entry_price = current_price
            self.entry_time = datetime.now()
            self.stop_loss_price = current_price * (1 - self.stop_loss_percentage / 100)
            self.take_profit_price = current_price * (1 + self.take_profit_percentage / 100)
            
            logger.info(f"📝 PAPER BUY: {self.symbol} at {current_price}")
            return True
            
        elif signal['signal'] == 'sell' and self.position == 'long':
            pnl = (current_price - self.entry_price) / self.entry_price * 100
            
            trade_record = {
                'entry_time': self.entry_time,
                'exit_time': datetime.now(),
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'pnl_percentage': pnl,
                'signal_confidence': signal.get('confidence', 0.0)
            }
            
            self.trades_history.append(trade_record)
            self.total_pnl += pnl
            
            if pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
            
            # Reset position
            self.position = None
            self.entry_price = None
            self.entry_time = None
            self.stop_loss_price = None
            self.take_profit_price = None
            
            logger.info(f"📝 PAPER SELL: {self.symbol} at {current_price} (PnL: {pnl:.2f}%)")
            return True
        
        return False
    
    def check_exit_conditions(self) -> bool:
        """Check if position should be closed due to stop loss or take profit"""
        if self.position is None:
            return False
        
        try:
            data = self.get_market_data(limit=1)
            if data is None or data.empty:
                return False
            
            current_price = data.iloc[-1]['close']
            
            # Check stop loss
            if current_price <= self.stop_loss_price:
                logger.info(f"🛑 Stop loss triggered at {current_price}")
                return self._execute_sell_order({'signal': 'sell', 'reason': 'stop_loss'})
            
            # Check take profit
            if current_price >= self.take_profit_price:
                logger.info(f"🎯 Take profit triggered at {current_price}")
                return self._execute_sell_order({'signal': 'sell', 'reason': 'take_profit'})
                
        except Exception as e:
            logger.error(f"❌ Error checking exit conditions: {e}")
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        total_trades = len(self.trades_history)
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': self.win_count,
            'losing_trades': self.loss_count,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'average_pnl': self.total_pnl / total_trades if total_trades > 0 else 0,
            'current_position': self.position,
            'entry_price': self.entry_price,
            'trades_history': self.trades_history[-10:]  # Last 10 trades
        }
    
    def reset_performance(self):
        """Reset performance tracking"""
        self.trades_history = []
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        logger.info("📊 Performance tracking reset")

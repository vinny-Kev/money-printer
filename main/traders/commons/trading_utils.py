"""
Trading Utilities
Common utility functions for trading operations
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TradingUtils:
    """Utility functions for trading operations"""
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty or len(df) < 20:
            logger.warning("⚠️ Insufficient data for technical indicators")
            return df
        
        df = df.copy()
        
        try:
            # Simple Moving Averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # Exponential Moving Averages
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_10'] = df['close'].ewm(span=10).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            
            # RSI (Relative Strength Index)
            df['rsi'] = TradingUtils._calculate_rsi(df['close'])
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_histogram'] = TradingUtils._calculate_macd(df['close'])
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = TradingUtils._calculate_bollinger_bands(df['close'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price change indicators
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(periods=5)
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Support and resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            logger.debug(f"✅ Calculated technical indicators for {len(df)} rows")
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"❌ Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            return macd, macd_signal, macd_histogram
        except Exception as e:
            logger.error(f"❌ Error calculating MACD: {e}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            return upper_band, sma, lower_band
        except Exception as e:
            logger.error(f"❌ Error calculating Bollinger Bands: {e}")
            mean_price = prices.mean()
            mean_series = pd.Series([mean_price] * len(prices), index=prices.index)
            return mean_series, mean_series, mean_series
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
        """
        Prepare feature matrix for ML models
        
        Args:
            df: DataFrame with market data and indicators
            feature_cols: List of feature column names to include
            
        Returns:
            DataFrame with selected features, cleaned and normalized
        """
        if df.empty:
            logger.warning("⚠️ Empty DataFrame provided for feature preparation")
            return pd.DataFrame()
        
        # Default feature columns
        if feature_cols is None:
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_5', 'sma_10', 'sma_20',
                'ema_5', 'ema_10', 'ema_20',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower',
                'volume_ratio', 'price_change', 'price_change_5', 'volatility'
            ]
        
        try:
            # Select available feature columns
            available_cols = [col for col in feature_cols if col in df.columns]
            if not available_cols:
                logger.error("❌ No feature columns found in DataFrame")
                return pd.DataFrame()
            
            features_df = df[available_cols].copy()
            
            # Handle missing values
            features_df = features_df.fillna(method='forward').fillna(method='backward')
            features_df = features_df.fillna(0)  # Fill any remaining NaN with 0
            
            # Remove infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(0)
            
            logger.debug(f"✅ Prepared {len(available_cols)} features for {len(features_df)} rows")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_position_size(balance: float, 
                              risk_percentage: float,
                              entry_price: float,
                              stop_loss_price: float,
                              max_position_size: float = None) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            balance: Available balance in USDT
            risk_percentage: Maximum risk per trade (%)
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            max_position_size: Maximum position size in USDT
            
        Returns:
            Position size in base currency units
        """
        try:
            if balance <= 0 or entry_price <= 0 or stop_loss_price <= 0:
                return 0.0
            
            # Calculate risk amount in USDT
            risk_amount = balance * (risk_percentage / 100)
            
            # Calculate price risk per unit
            price_risk = abs(entry_price - stop_loss_price)
            if price_risk == 0:
                return 0.0
            
            # Calculate position size
            position_size_usdt = risk_amount / (price_risk / entry_price)
            
            # Apply maximum position size limit
            if max_position_size and position_size_usdt > max_position_size:
                position_size_usdt = max_position_size
            
            # Convert to base currency units
            position_size_units = position_size_usdt / entry_price
            
            logger.debug(f"💰 Position size calculation: {position_size_usdt:.2f} USDT = {position_size_units:.6f} units")
            
            return position_size_units
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.0
    
    @staticmethod
    def validate_signal(signal: Dict[str, Any]) -> bool:
        """
        Validate trading signal format and content
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            bool: True if signal is valid
        """
        try:
            # Check required fields
            required_fields = ['signal', 'confidence']
            for field in required_fields:
                if field not in signal:
                    logger.error(f"❌ Missing required field in signal: {field}")
                    return False
            
            # Validate signal value
            valid_signals = ['buy', 'sell', 'hold']
            if signal['signal'] not in valid_signals:
                logger.error(f"❌ Invalid signal value: {signal['signal']}")
                return False
            
            # Validate confidence
            confidence = signal['confidence']
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                logger.error(f"❌ Invalid confidence value: {confidence}")
                return False
            
            logger.debug(f"✅ Signal validation passed: {signal['signal']} (confidence: {confidence:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating signal: {e}")
            return False
    
    @staticmethod
    def format_price(price: float, symbol: str = 'BTCUSDT') -> str:
        """Format price with appropriate decimal places for the symbol"""
        try:
            if 'USDT' in symbol:
                return f"{price:.4f}"
            else:
                return f"{price:.8f}"
        except:
            return f"{price:.4f}"
    
    @staticmethod
    def format_percentage(percentage: float) -> str:
        """Format percentage with sign and color coding"""
        try:
            sign = "+" if percentage >= 0 else ""
            return f"{sign}{percentage:.2f}%"
        except:
            return "0.00%"
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio for returns
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        try:
            if returns.empty or returns.std() == 0:
                return 0.0
            
            excess_returns = returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / returns.std()
            
            # Annualize (assuming daily returns)
            sharpe_ratio_annual = sharpe_ratio * np.sqrt(365)
            
            return sharpe_ratio_annual
            
        except Exception as e:
            logger.error(f"❌ Error calculating Sharpe ratio: {e}")
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown from equity curve
        
        Args:
            equity_curve: Series of cumulative returns/equity values
            
        Returns:
            Maximum drawdown percentage
        """
        try:
            if equity_curve.empty:
                return 0.0
            
            # Calculate running maximum
            peak = equity_curve.expanding().max()
            
            # Calculate drawdown
            drawdown = (equity_curve - peak) / peak
            
            # Return maximum drawdown as positive percentage
            max_drawdown = abs(drawdown.min()) * 100
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"❌ Error calculating maximum drawdown: {e}")
            return 0.0

"""
Feature Engineer
Advanced feature engineering for machine learning models
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for financial time series"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_cache = {}
        logger.info("⚙️ FeatureEngineer initialized")
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            data: Input market data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if data.empty:
                return data
            
            logger.info(f"⚙️ Engineering features for {len(data)} rows")
            
            # Start with original data
            featured_data = data.copy()
            
            # Apply feature engineering steps
            featured_data = self._add_price_features(featured_data)
            featured_data = self._add_volume_features(featured_data)
            featured_data = self._add_technical_indicators(featured_data)
            featured_data = self._add_statistical_features(featured_data)
            featured_data = self._add_time_features(featured_data)
            featured_data = self._add_lag_features(featured_data)
            featured_data = self._add_volatility_features(featured_data)
            featured_data = self._add_momentum_features(featured_data)
            
            # Clean up any remaining NaN or infinite values
            featured_data = self._clean_features(featured_data)
            
            feature_count = len(featured_data.columns) - len(data.columns)
            logger.info(f"✅ Feature engineering completed: {feature_count} new features added")
            
            return featured_data
            
        except Exception as e:
            logger.error(f"❌ Error in feature engineering: {e}")
            return data
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                return data
            
            # Basic price relationships
            data['price_range'] = (data['high'] - data['low']) / data['close']
            data['body_size'] = abs(data['close'] - data['open']) / data['close']
            data['upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['close']
            data['lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['close']
            
            # Price position within range
            data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
            data['open_position'] = (data['open'] - data['low']) / (data['high'] - data['low'])
            
            # Price gaps
            data['gap'] = data.groupby('symbol')['open'].shift(0) / data.groupby('symbol')['close'].shift(1) - 1
            
            # True range
            data['prev_close'] = data.groupby('symbol')['close'].shift(1)
            data['true_range'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['prev_close']),
                    abs(data['low'] - data['prev_close'])
                )
            ) / data['close']
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding price features: {e}")
            return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            if 'volume' not in data.columns:
                return data
            
            # Volume moving averages
            for window in [5, 10, 20]:
                data[f'volume_ma_{window}'] = data.groupby('symbol')['volume'].rolling(window).mean().reset_index(0, drop=True)
                data[f'volume_ratio_{window}'] = data['volume'] / data[f'volume_ma_{window}']
            
            # Volume profile
            data['volume_std_20'] = data.groupby('symbol')['volume'].rolling(20).std().reset_index(0, drop=True)
            data['volume_zscore'] = (data['volume'] - data['volume_ma_20']) / data['volume_std_20']
            
            # Price-Volume features
            if 'close' in data.columns:
                data['price_change'] = data.groupby('symbol')['close'].pct_change()
                data['volume_price_trend'] = data['volume'] * data['price_change']
                
                # On-Balance Volume (OBV)
                data['obv_change'] = data['volume'] * np.sign(data['price_change'])
                data['obv'] = data.groupby('symbol')['obv_change'].cumsum()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding volume features: {e}")
            return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        try:
            if 'close' not in data.columns:
                return data
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                data[f'sma_{window}'] = data.groupby('symbol')['close'].rolling(window).mean().reset_index(0, drop=True)
                data[f'ema_{window}'] = data.groupby('symbol')['close'].ewm(span=window).mean().reset_index(0, drop=True)
                
                # Price vs moving average
                data[f'price_vs_sma_{window}'] = data['close'] / data[f'sma_{window}'] - 1
                data[f'price_vs_ema_{window}'] = data['close'] / data[f'ema_{window}'] - 1
            
            # RSI
            data['rsi'] = data.groupby('symbol').apply(lambda x: self._calculate_rsi(x['close'])).reset_index(0, drop=True)
            
            # MACD
            grouped = data.groupby('symbol')
            for symbol, group in grouped:
                macd, signal, histogram = self._calculate_macd(group['close'])
                data.loc[data['symbol'] == symbol, 'macd'] = macd.values
                data.loc[data['symbol'] == symbol, 'macd_signal'] = signal.values
                data.loc[data['symbol'] == symbol, 'macd_histogram'] = histogram.values
            
            # Bollinger Bands
            for window in [20]:
                data[f'bb_middle_{window}'] = data.groupby('symbol')['close'].rolling(window).mean().reset_index(0, drop=True)
                data[f'bb_std_{window}'] = data.groupby('symbol')['close'].rolling(window).std().reset_index(0, drop=True)
                data[f'bb_upper_{window}'] = data[f'bb_middle_{window}'] + 2 * data[f'bb_std_{window}']
                data[f'bb_lower_{window}'] = data[f'bb_middle_{window}'] - 2 * data[f'bb_std_{window}']
                
                # Bollinger Band position
                data[f'bb_position_{window}'] = (data['close'] - data[f'bb_lower_{window}']) / (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}'])
                data[f'bb_width_{window}'] = (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}']) / data[f'bb_middle_{window}']
            
            # Stochastic Oscillator
            if all(col in data.columns for col in ['high', 'low']):
                for window in [14]:
                    data[f'lowest_low_{window}'] = data.groupby('symbol')['low'].rolling(window).min().reset_index(0, drop=True)
                    data[f'highest_high_{window}'] = data.groupby('symbol')['high'].rolling(window).max().reset_index(0, drop=True)
                    data[f'stoch_k_{window}'] = (data['close'] - data[f'lowest_low_{window}']) / (data[f'highest_high_{window}'] - data[f'lowest_low_{window}']) * 100
                    data[f'stoch_d_{window}'] = data.groupby('symbol')[f'stoch_k_{window}'].rolling(3).mean().reset_index(0, drop=True)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding technical indicators: {e}")
            return data
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        try:
            if 'close' not in data.columns:
                return data
            
            # Rolling statistics
            for window in [10, 20, 50]:
                # Standard deviation
                data[f'price_std_{window}'] = data.groupby('symbol')['close'].rolling(window).std().reset_index(0, drop=True)
                
                # Skewness
                data[f'price_skew_{window}'] = data.groupby('symbol')['close'].rolling(window).skew().reset_index(0, drop=True)
                
                # Kurtosis
                data[f'price_kurt_{window}'] = data.groupby('symbol')['close'].rolling(window).kurt().reset_index(0, drop=True)
                
                # Quantiles
                data[f'price_q25_{window}'] = data.groupby('symbol')['close'].rolling(window).quantile(0.25).reset_index(0, drop=True)
                data[f'price_q75_{window}'] = data.groupby('symbol')['close'].rolling(window).quantile(0.75).reset_index(0, drop=True)
                
                # Position within distribution
                data[f'price_percentile_{window}'] = data.groupby('symbol')['close'].rolling(window).rank(pct=True).reset_index(0, drop=True)
            
            # Price returns distribution
            if 'price_change' in data.columns:
                for window in [20, 50]:
                    data[f'returns_std_{window}'] = data.groupby('symbol')['price_change'].rolling(window).std().reset_index(0, drop=True)
                    data[f'returns_skew_{window}'] = data.groupby('symbol')['price_change'].rolling(window).skew().reset_index(0, drop=True)
                    data[f'returns_kurt_{window}'] = data.groupby('symbol')['price_change'].rolling(window).kurt().reset_index(0, drop=True)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding statistical features: {e}")
            return data
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            if 'timestamp' not in data.columns:
                return data
            
            # Ensure timestamp is datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Time components
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['day_of_month'] = data['timestamp'].dt.day
            data['month'] = data['timestamp'].dt.month
            data['quarter'] = data['timestamp'].dt.quarter
            
            # Time cyclical features
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            # Market session features
            data['is_weekend'] = data['day_of_week'].isin([5, 6])
            data['is_market_hours'] = data['hour'].between(9, 16)  # Approximate market hours
            data['is_asian_session'] = data['hour'].between(0, 8)
            data['is_european_session'] = data['hour'].between(8, 16)
            data['is_american_session'] = data['hour'].between(16, 24)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding time features: {e}")
            return data
    
    def _add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        try:
            if 'close' not in data.columns:
                return data
            
            # Price lags
            for lag in [1, 2, 3, 5, 10]:
                data[f'close_lag_{lag}'] = data.groupby('symbol')['close'].shift(lag)
                data[f'close_change_lag_{lag}'] = data.groupby('symbol')['close'].pct_change(lag)
            
            # Volume lags
            if 'volume' in data.columns:
                for lag in [1, 2, 3]:
                    data[f'volume_lag_{lag}'] = data.groupby('symbol')['volume'].shift(lag)
                    data[f'volume_change_lag_{lag}'] = data.groupby('symbol')['volume'].pct_change(lag)
            
            # Technical indicator lags
            if 'rsi' in data.columns:
                for lag in [1, 2]:
                    data[f'rsi_lag_{lag}'] = data.groupby('symbol')['rsi'].shift(lag)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding lag features: {e}")
            return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
            if 'close' not in data.columns:
                return data
            
            # Price change if not already calculated
            if 'price_change' not in data.columns:
                data['price_change'] = data.groupby('symbol')['close'].pct_change()
            
            # Historical volatility
            for window in [10, 20, 50]:
                data[f'volatility_{window}'] = data.groupby('symbol')['price_change'].rolling(window).std().reset_index(0, drop=True)
                data[f'volatility_ma_{window}'] = data.groupby('symbol')[f'volatility_{window}'].rolling(window).mean().reset_index(0, drop=True)
            
            # GARCH-like features
            data['price_change_squared'] = data['price_change'] ** 2
            for window in [10, 20]:
                data[f'garch_variance_{window}'] = data.groupby('symbol')['price_change_squared'].rolling(window).mean().reset_index(0, drop=True)
            
            # True Range based volatility
            if 'true_range' in data.columns:
                for window in [14, 20]:
                    data[f'atr_{window}'] = data.groupby('symbol')['true_range'].rolling(window).mean().reset_index(0, drop=True)
                    data[f'volatility_ratio_{window}'] = data['true_range'] / data[f'atr_{window}']
            
            # Parkinson volatility (using high-low)
            if all(col in data.columns for col in ['high', 'low']):
                data['parkinson_vol'] = np.sqrt(0.25 * np.log(data['high'] / data['low']) ** 2)
                for window in [20]:
                    data[f'parkinson_vol_ma_{window}'] = data.groupby('symbol')['parkinson_vol'].rolling(window).mean().reset_index(0, drop=True)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding volatility features: {e}")
            return data
    
    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        try:
            if 'close' not in data.columns:
                return data
            
            # Rate of Change (ROC)
            for period in [5, 10, 20]:
                data[f'roc_{period}'] = data.groupby('symbol')['close'].pct_change(period)
            
            # Momentum oscillator
            for period in [10, 20]:
                data[f'momentum_{period}'] = data['close'] / data.groupby('symbol')['close'].shift(period) - 1
            
            # Price acceleration
            if 'price_change' not in data.columns:
                data['price_change'] = data.groupby('symbol')['close'].pct_change()
            
            data['price_acceleration'] = data.groupby('symbol')['price_change'].diff()
            
            # Trend strength
            for window in [10, 20]:
                # Count of positive vs negative changes
                data[f'positive_changes_{window}'] = (
                    (data.groupby('symbol')['price_change'].rolling(window).apply(lambda x: (x > 0).sum())) / window
                ).reset_index(0, drop=True)
                
                # Trend consistency
                data[f'trend_consistency_{window}'] = (
                    data.groupby('symbol')['price_change'].rolling(window).apply(
                        lambda x: 1 - (abs(x > 0).sum() - abs(x < 0).sum()) / window
                    )
                ).reset_index(0, drop=True)
            
            # Williams %R
            if all(col in data.columns for col in ['high', 'low']):
                for window in [14]:
                    highest_high = data.groupby('symbol')['high'].rolling(window).max().reset_index(0, drop=True)
                    lowest_low = data.groupby('symbol')['low'].rolling(window).min().reset_index(0, drop=True)
                    data[f'williams_r_{window}'] = (highest_high - data['close']) / (highest_high - lowest_low) * -100
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding momentum features: {e}")
            return data
    
    def _clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate engineered features"""
        try:
            # Replace infinite values with NaN
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values within groups
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                data[col] = data.groupby('symbol')[col].fillna(method='ffill')
            
            # Backward fill remaining NaN values
            for col in numeric_cols:
                data[col] = data.groupby('symbol')[col].fillna(method='bfill')
            
            # Fill any remaining NaN with 0
            data = data.fillna(0)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error cleaning features: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
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
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
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

"""
Technical Indicators Module - Pure Python Implementation
Uses the 'ta' library (by bukosabino) for cleaner, easier-to-deploy indicators.
Supports all core indicators: RSI, MACD, Bollinger Bands, EMA/SMA, ATR, and Stoch RSI.
"""

import pandas as pd
import ta
import numpy as np
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Technical Indicators calculator with configurable parameters.
    Pure Python implementation using pandas_ta.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with optional configuration for indicator parameters.
        
        Args:
            config: Dictionary with indicator parameters
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Default configuration for all indicators."""
        return {
            'rsi': {'length': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bb': {'length': 20, 'std': 2},
            'ema': {'lengths': [9, 21, 50]},
            'sma': {'lengths': [20, 50, 200]},
            'atr': {'length': 14},
            'stoch_rsi': {'length': 14, 'rsi_length': 14, 'k': 3, 'd': 3},
            'volume': {'window': 10}
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        try:
            df = df.copy()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate each indicator
            df = self._calculate_rsi(df)
            df = self._calculate_macd(df)
            df = self._calculate_bollinger_bands(df)
            df = self._calculate_moving_averages(df)
            df = self._calculate_atr(df)
            df = self._calculate_stoch_rsi(df)
            df = self._calculate_volume_indicators(df)
            df = self._calculate_price_features(df)
            
            # Forward fill any remaining NaN values
            df = df.ffill().fillna(0)
            
            logger.info(f"Successfully calculated indicators for {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI)."""
        try:
            length = self.config['rsi']['length']
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=length).rsi()
            return df
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            df['rsi'] = 50  # Neutral RSI value as fallback
            return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            fast = self.config['macd']['fast']
            slow = self.config['macd']['slow']
            signal = self.config['macd']['signal']
            
            macd_indicator = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
            
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_hist'] = macd_indicator.macd_diff()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            # Fallback values
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_hist'] = 0
            return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            length = self.config['bb']['length']
            std = self.config['bb']['std']
            
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=length, window_dev=std)
            
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            # Fallback values
            df['bb_upper'] = df['close'] * 1.02
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close'] * 0.98
            df['bb_width'] = 0.04
            df['bb_percent'] = 0.5
            return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exponential and Simple Moving Averages."""
        try:
            # EMA
            for length in self.config['ema']['lengths']:
                df[f'ema_{length}'] = ta.trend.EMAIndicator(df['close'], window=length).ema_indicator()
            
            # SMA
            for length in self.config['sma']['lengths']:
                df[f'sma_{length}'] = ta.trend.SMAIndicator(df['close'], window=length).sma_indicator()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            # Fallback - use close price
            for length in self.config['ema']['lengths']:
                df[f'ema_{length}'] = df['close']
            for length in self.config['sma']['lengths']:
                df[f'sma_{length}'] = df['close']
            return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range (ATR)."""
        try:
            length = self.config['atr']['length']
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=length).average_true_range()
            df['atr_percent'] = df['atr'] / df['close'] * 100
            return df
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            # Fallback - calculate simple range
            df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
            df['atr_percent'] = df['atr'] / df['close'] * 100
            return df
    
    def _calculate_stoch_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic RSI."""
        try:
            length = self.config['stoch_rsi']['length']
            rsi_length = self.config['stoch_rsi']['rsi_length']
            k = self.config['stoch_rsi']['k']
            d = self.config['stoch_rsi']['d']
            
            stoch_rsi_indicator = ta.momentum.StochRSIIndicator(df['close'], window=length, smooth1=k, smooth2=d)
            
            df['stoch_rsi_k'] = stoch_rsi_indicator.stochrsi_k()
            df['stoch_rsi_d'] = stoch_rsi_indicator.stochrsi_d()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Stochastic RSI: {e}")
            # Fallback values
            df['stoch_rsi_k'] = 50
            df['stoch_rsi_d'] = 50
            return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        try:
            window = self.config['volume']['window']
            
            # Volume indicators
            df['volume_change'] = df['volume'].pct_change().fillna(0)
            df['volume_ma'] = df['volume'].rolling(window=window).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Volume Z-score
            df['volume_zscore'] = (
                (df['volume'] - df['volume'].rolling(window=window).mean()) /
                df['volume'].rolling(window=window).std()
            ).fillna(0)
            
            # On-Balance Volume (OBV)
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            # Fallback values
            df['volume_change'] = 0
            df['volume_ma'] = df['volume']
            df['volume_ratio'] = 1
            df['volume_zscore'] = 0
            df['obv'] = df['volume'].cumsum()
            return df
    
    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional price-based features."""
        try:
            # Price correlations
            df['price_volume_corr'] = df['close'].rolling(window=10).corr(df['volume']).fillna(0)
            
            # Price differences
            df['open_close_diff'] = df['open'] - df['close']
            df['high_low_diff'] = df['high'] - df['low']
            df['price_change'] = df['close'].pct_change().fillna(0)
            
            # Price position within candle
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['close_position'] = df['close_position'].fillna(0.5)
            
            # Momentum indicators
            df['roc'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
            df['momentum'] = df['close'].diff(10)  # Simple momentum calculation
            
            return df
        except Exception as e:
            logger.error(f"Error calculating price features: {e}")
            # Fallback values
            df['price_volume_corr'] = 0
            df['open_close_diff'] = 0
            df['high_low_diff'] = df['high'] - df['low']
            df['price_change'] = 0
            df['close_position'] = 0.5
            df['roc'] = 0
            df['momentum'] = 0
            return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be calculated."""
        features = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            'atr', 'atr_percent',
            'stoch_rsi_k', 'stoch_rsi_d',
            'volume_change', 'volume_ma', 'volume_ratio', 'volume_zscore', 'obv',
            'price_volume_corr', 'open_close_diff', 'high_low_diff', 
            'price_change', 'close_position', 'roc', 'momentum'
        ]
        
        # Add EMA features
        for length in self.config['ema']['lengths']:
            features.append(f'ema_{length}')
        
        # Add SMA features
        for length in self.config['sma']['lengths']:
            features.append(f'sma_{length}')
        
        return features
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame has all required indicators."""
        try:
            expected_features = self.get_feature_names()
            missing_features = [f for f in expected_features if f not in df.columns]
            
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                return False
            
            # Check for excessive NaN values
            nan_counts = df[expected_features].isnull().sum()
            excessive_nans = nan_counts[nan_counts > len(df) * 0.1]
            
            if not excessive_nans.empty:
                logger.warning(f"Features with >10% NaN values: {excessive_nans.to_dict()}")
                return False
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False


def create_indicator_calculator(config: Optional[Dict] = None) -> TechnicalIndicators:
    """
    Factory function to create a technical indicators calculator.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        TechnicalIndicators instance
    """
    return TechnicalIndicators(config)


# Legacy compatibility function
def calculate_rsi_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    Calculates RSI, MACD, and other features using the new indicators module.
    """
    try:
        indicators = TechnicalIndicators()
        return indicators.calculate_all_indicators(df)
    except Exception as e:
        logger.error(f"Error in legacy calculate_rsi_macd function: {e}")
        return df


if __name__ == "__main__":
    # Example usage and testing
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, 100),
        'symbol': 'BTCUSDT'
    })
    
    # Calculate high/low based on open/close
    sample_data['close'] = sample_data['open'] + np.random.randn(100) * 0.5
    sample_data['high'] = np.maximum(sample_data['open'], sample_data['close']) + np.abs(np.random.randn(100) * 0.2)
    sample_data['low'] = np.minimum(sample_data['open'], sample_data['close']) - np.abs(np.random.randn(100) * 0.2)
    
    # Test indicators
    indicators = TechnicalIndicators()
    result = indicators.calculate_all_indicators(sample_data)
    
    print("Sample output with indicators:")
    print(result[['timestamp', 'close', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_9', 'atr']].tail())
    print(f"\nTotal features: {len(indicators.get_feature_names())}")
    print(f"Validation passed: {indicators.validate_data(result)}")

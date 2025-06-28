"""
Data Processor
Utilities for data collection, cleaning, and preprocessing
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataProcessor:
    """Utilities for data processing and cleaning"""
    
    def __init__(self):
        """Initialize data processor"""
        self.processed_data_cache = {}
        logger.info("🔧 DataProcessor initialized")
    
    def clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize market data
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            if data.empty:
                return data
            
            logger.info(f"🧹 Cleaning {len(data)} rows of market data")
            
            cleaned_data = data.copy()
            
            # Standardize column names
            cleaned_data = self._standardize_columns(cleaned_data)
            
            # Handle timestamps
            cleaned_data = self._process_timestamps(cleaned_data)
            
            # Clean numeric columns
            cleaned_data = self._clean_numeric_columns(cleaned_data)
            
            # Remove invalid rows
            cleaned_data = self._remove_invalid_rows(cleaned_data)
            
            # Sort by timestamp and symbol
            if 'timestamp' in cleaned_data.columns:
                sort_cols = ['timestamp']
                if 'symbol' in cleaned_data.columns:
                    sort_cols.append('symbol')
                cleaned_data = cleaned_data.sort_values(sort_cols).reset_index(drop=True)
            
            rows_removed = len(data) - len(cleaned_data)
            if rows_removed > 0:
                logger.info(f"🗑️ Removed {rows_removed} invalid rows")
            
            logger.info(f"✅ Data cleaning completed: {len(cleaned_data)} rows remaining")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"❌ Error cleaning market data: {e}")
            return data
    
    def aggregate_multi_timeframe_data(self, 
                                     data: pd.DataFrame,
                                     base_timeframe: str = '1m',
                                     target_timeframes: List[str] = None) -> pd.DataFrame:
        """
        Aggregate data to multiple timeframes
        
        Args:
            data: Minute-level market data
            base_timeframe: Base timeframe of input data
            target_timeframes: List of target timeframes to create
            
        Returns:
            DataFrame with multi-timeframe features
        """
        try:
            if target_timeframes is None:
                target_timeframes = ['5m', '15m', '1h', '4h']
            
            logger.info(f"📊 Aggregating data to timeframes: {target_timeframes}")
            
            if data.empty or 'timestamp' not in data.columns:
                return data
            
            # Group by symbol for processing
            result_dfs = []
            
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
                
                # Create aggregated features for each timeframe
                for timeframe in target_timeframes:
                    aggregated = self._aggregate_to_timeframe(symbol_data, timeframe)
                    
                    # Merge back to original data
                    symbol_data = pd.merge_asof(
                        symbol_data.sort_values('timestamp'),
                        aggregated.sort_values('timestamp'),
                        on='timestamp',
                        suffixes=('', f'_{timeframe}'),
                        direction='backward'
                    )
                
                result_dfs.append(symbol_data)
            
            result = pd.concat(result_dfs, ignore_index=True)
            
            logger.info(f"✅ Multi-timeframe aggregation completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in multi-timeframe aggregation: {e}")
            return data
    
    def create_rolling_features(self, 
                               data: pd.DataFrame,
                               windows: List[int] = None,
                               columns: List[str] = None) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            data: Input DataFrame
            windows: List of rolling window sizes
            columns: List of columns to create rolling features for
            
        Returns:
            DataFrame with rolling features added
        """
        try:
            if windows is None:
                windows = [5, 10, 20, 50]
            
            if columns is None:
                columns = ['close', 'volume']
            
            logger.info(f"📈 Creating rolling features: windows={windows}, columns={columns}")
            
            if data.empty:
                return data
            
            result_data = data.copy()
            
            # Group by symbol for rolling calculations
            for symbol in result_data['symbol'].unique():
                symbol_mask = result_data['symbol'] == symbol
                symbol_data = result_data[symbol_mask].copy()
                
                for col in columns:
                    if col in symbol_data.columns:
                        for window in windows:
                            # Rolling mean
                            result_data.loc[symbol_mask, f'{col}_ma_{window}'] = \
                                symbol_data[col].rolling(window=window, min_periods=1).mean()
                            
                            # Rolling std
                            result_data.loc[symbol_mask, f'{col}_std_{window}'] = \
                                symbol_data[col].rolling(window=window, min_periods=1).std()
                            
                            # Rolling min/max
                            result_data.loc[symbol_mask, f'{col}_min_{window}'] = \
                                symbol_data[col].rolling(window=window, min_periods=1).min()
                            result_data.loc[symbol_mask, f'{col}_max_{window}'] = \
                                symbol_data[col].rolling(window=window, min_periods=1).max()
                            
                            # Relative position in range
                            if col == 'close':
                                rolling_min = symbol_data[col].rolling(window=window, min_periods=1).min()
                                rolling_max = symbol_data[col].rolling(window=window, min_periods=1).max()
                                range_size = rolling_max - rolling_min
                                range_size = range_size.replace(0, np.nan)  # Avoid division by zero
                                
                                result_data.loc[symbol_mask, f'{col}_position_{window}'] = \
                                    (symbol_data[col] - rolling_min) / range_size
            
            logger.info(f"✅ Rolling features created")
            return result_data
            
        except Exception as e:
            logger.error(f"❌ Error creating rolling features: {e}")
            return data
    
    def detect_outliers(self, 
                       data: pd.DataFrame,
                       columns: List[str] = None,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and optionally remove outliers
        
        Args:
            data: Input DataFrame
            columns: Columns to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier information added
        """
        try:
            if columns is None:
                columns = ['close', 'volume', 'high', 'low']
            
            logger.info(f"🔍 Detecting outliers using {method} method")
            
            result_data = data.copy()
            outlier_columns = []
            
            for col in columns:
                if col in result_data.columns:
                    outlier_col = f'{col}_outlier'
                    outlier_columns.append(outlier_col)
                    
                    if method == 'iqr':
                        Q1 = result_data[col].quantile(0.25)
                        Q3 = result_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        
                        result_data[outlier_col] = (
                            (result_data[col] < lower_bound) | 
                            (result_data[col] > upper_bound)
                        )
                        
                    elif method == 'zscore':
                        z_scores = np.abs((result_data[col] - result_data[col].mean()) / result_data[col].std())
                        result_data[outlier_col] = z_scores > threshold
            
            # Summary outlier flag
            if outlier_columns:
                result_data['is_outlier'] = result_data[outlier_columns].any(axis=1)
                outlier_count = result_data['is_outlier'].sum()
                
                logger.info(f"🚨 Found {outlier_count} outlier rows ({outlier_count/len(result_data)*100:.1f}%)")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ Error detecting outliers: {e}")
            return data
    
    def fill_missing_data(self, 
                         data: pd.DataFrame,
                         method: str = 'forward',
                         limit: int = None) -> pd.DataFrame:
        """
        Fill missing data using specified method
        
        Args:
            data: Input DataFrame
            method: Fill method ('forward', 'backward', 'interpolate', 'mean')
            limit: Maximum number of consecutive fills
            
        Returns:
            DataFrame with missing data filled
        """
        try:
            logger.info(f"🔧 Filling missing data using {method} method")
            
            result_data = data.copy()
            
            # Identify numeric columns
            numeric_cols = result_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                missing_before = result_data[col].isnull().sum()
                
                if missing_before > 0:
                    if method == 'forward':
                        result_data[col] = result_data[col].fillna(method='ffill', limit=limit)
                    elif method == 'backward':
                        result_data[col] = result_data[col].fillna(method='bfill', limit=limit)
                    elif method == 'interpolate':
                        result_data[col] = result_data[col].interpolate(limit=limit)
                    elif method == 'mean':
                        mean_value = result_data[col].mean()
                        result_data[col] = result_data[col].fillna(mean_value)
                    
                    missing_after = result_data[col].isnull().sum()
                    filled = missing_before - missing_after
                    
                    if filled > 0:
                        logger.info(f"   {col}: filled {filled} missing values")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ Error filling missing data: {e}")
            return data
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        try:
            # Common column mappings
            column_mappings = {
                'open_time': 'timestamp',
                'close_time': 'close_time',
                'open_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'close_price': 'close',
                'base_volume': 'volume',
                'quote_volume': 'quote_asset_volume'
            }
            
            # Apply mappings
            data = data.rename(columns=column_mappings)
            
            # Ensure standard numeric columns are present
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    logger.warning(f"⚠️ Missing required column: {col}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error standardizing columns: {e}")
            return data
    
    def _process_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize timestamps"""
        try:
            if 'timestamp' in data.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    # Try different timestamp formats
                    try:
                        # Unix timestamp in milliseconds
                        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                    except:
                        try:
                            # Unix timestamp in seconds
                            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
                        except:
                            # ISO format or other
                            data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Add time-based features
                data['hour'] = data['timestamp'].dt.hour
                data['day_of_week'] = data['timestamp'].dt.dayofweek
                data['is_weekend'] = data['day_of_week'].isin([5, 6])
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error processing timestamps: {e}")
            return data
    
    def _clean_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns"""
        try:
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            
            for col in numeric_cols:
                if col in data.columns:
                    # Convert to numeric, coercing errors to NaN
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    
                    # Remove infinite values
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Remove negative prices (except for returns/changes)
                    if col in ['open', 'high', 'low', 'close']:
                        data = data[data[col] > 0]
                    
                    # Remove zero volume (often indicates bad data)
                    if col == 'volume':
                        data = data[data[col] > 0]
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error cleaning numeric columns: {e}")
            return data
    
    def _remove_invalid_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid rows"""
        try:
            initial_rows = len(data)
            
            # Remove rows with all NaN values
            data = data.dropna(how='all')
            
            # Remove rows where high < low (impossible)
            if 'high' in data.columns and 'low' in data.columns:
                data = data[data['high'] >= data['low']]
            
            # Remove rows where close is outside high/low range
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                data = data[
                    (data['close'] >= data['low']) & 
                    (data['close'] <= data['high']) &
                    (data['open'] >= data['low']) & 
                    (data['open'] <= data['high'])
                ]
            
            removed_rows = initial_rows - len(data)
            if removed_rows > 0:
                logger.info(f"🗑️ Removed {removed_rows} invalid rows")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error removing invalid rows: {e}")
            return data
    
    def _aggregate_to_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate data to specified timeframe"""
        try:
            # Parse timeframe
            timeframe_mapping = {
                '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
                '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
                '1d': '1D', '1w': '1W'
            }
            
            freq = timeframe_mapping.get(timeframe, timeframe)
            
            # Set timestamp as index for resampling
            data_indexed = data.set_index('timestamp')
            
            # Define aggregation functions
            agg_funcs = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Apply only to available columns
            available_agg_funcs = {
                col: func for col, func in agg_funcs.items() 
                if col in data_indexed.columns
            }
            
            # Resample and aggregate
            aggregated = data_indexed.resample(freq).agg(available_agg_funcs)
            
            # Reset index to get timestamp back as column
            aggregated = aggregated.reset_index()
            
            # Add symbol back if it was in original data
            if 'symbol' in data.columns:
                aggregated['symbol'] = data['symbol'].iloc[0]
            
            return aggregated
            
        except Exception as e:
            logger.error(f"❌ Error aggregating to {timeframe}: {e}")
            return pd.DataFrame()

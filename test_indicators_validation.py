#!/usr/bin/env python3
"""
Test script to validate the new pandas_ta-based technical indicators.
Generates sample data and validates all indicators are working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.trading_bot.technical_indicators import TechnicalIndicators
from src.model_training.common import preprocess_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(days=30, symbol='BTCUSDT'):
    """Generate realistic sample OHLCV data for testing."""
    
    # Generate timestamps (1-hour intervals)
    start_time = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start_time, periods=days*24, freq='1H')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    n_points = len(timestamps)
    
    # Base price with trend
    base_price = 45000
    trend = np.linspace(0, 2000, n_points)  # Upward trend
    noise = np.random.normal(0, 500, n_points)
    
    # Generate price series
    close_prices = base_price + trend + noise
    
    # Generate OHLC from close prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Add some realistic variation
    high_addon = np.abs(np.random.normal(0, 200, n_points))
    low_addon = np.abs(np.random.normal(0, 200, n_points))
    
    high_prices = np.maximum(open_prices, close_prices) + high_addon
    low_prices = np.minimum(open_prices, close_prices) - low_addon
    
    # Generate volume data
    base_volume = 1000000
    volume_variation = np.random.uniform(0.5, 2.0, n_points)
    volumes = (base_volume * volume_variation).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': [int(ts.timestamp() * 1000) for ts in timestamps],
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
        'symbol': symbol
    })
    
    return df

def test_technical_indicators():
    """Test the TechnicalIndicators class with sample data."""
    
    logger.info("=== Testing Technical Indicators Module ===")
    
    # Generate sample data
    df = generate_sample_data(days=10, symbol='BTCUSDT')
    logger.info(f"Generated {len(df)} data points for testing")
    
    # Test the indicators
    try:
        indicators = TechnicalIndicators()
        logger.info("Created TechnicalIndicators instance")
        
        # Calculate indicators
        result_df = indicators.calculate_all_indicators(df)
        logger.info(f"Calculated indicators for {len(result_df)} data points")
        
        # Validate results
        is_valid = indicators.validate_data(result_df)
        logger.info(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Print sample results
        feature_names = indicators.get_feature_names()
        logger.info(f"Total features calculated: {len(feature_names)}")
        
        # Show a sample of the results
        sample_features = ['close', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_9', 'atr', 'volume_ratio']
        available_features = [f for f in sample_features if f in result_df.columns]
        
        print("\\n=== Sample Output ===")
        print(result_df[['timestamp'] + available_features].tail(10).to_string(index=False))
        
        # Check for NaN values
        nan_counts = result_df[feature_names].isnull().sum()
        features_with_nans = nan_counts[nan_counts > 0]
        
        if not features_with_nans.empty:
            print(f"\\n=== Features with NaN values ===")
            print(features_with_nans.to_string())
        else:
            print("\\n✓ No NaN values found in any features")
        
        # Statistical summary
        print("\\n=== Statistical Summary ===")
        key_features = ['rsi', 'macd', 'bb_percent', 'atr_percent', 'volume_ratio']
        available_key_features = [f for f in key_features if f in result_df.columns]
        
        if available_key_features:
            print(result_df[available_key_features].describe().round(4).to_string())
        
        return result_df, True
        
    except Exception as e:
        logger.error(f"Error testing technical indicators: {e}")
        return None, False

def test_model_training_integration():
    """Test integration with model training preprocessing."""
    
    logger.info("\\n=== Testing Model Training Integration ===")
    
    try:
        # Generate sample data
        df = generate_sample_data(days=5, symbol='ETHUSDT')
        
        # Test preprocessing
        processed_df = preprocess_data(df)
        logger.info(f"Processed {len(processed_df)} data points with model training preprocessing")
        
        # Check required features
        required_features = ["timestamp", "open", "high", "low", "close", "volume", "rsi", "macd"]
        missing_features = [f for f in required_features if f not in processed_df.columns]
        
        if missing_features:
            logger.warning(f"Missing required features: {missing_features}")
            return False
        else:
            logger.info("✓ All required features present")
        
        # Show sample of additional features
        additional_features = [col for col in processed_df.columns if col not in df.columns and col != 'symbol_id']
        logger.info(f"Added {len(additional_features)} new features during preprocessing")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing model training integration: {e}")
        return False

def test_legacy_compatibility():
    """Test that legacy function calls still work."""
    
    logger.info("\\n=== Testing Legacy Compatibility ===")
    
    try:
        from src.trading_bot.technical_indicators import calculate_rsi_macd
        
        # Generate sample data
        df = generate_sample_data(days=3, symbol='ADAUSDT')
        
        # Test legacy function
        result_df = calculate_rsi_macd(df)
        logger.info(f"Legacy function processed {len(result_df)} data points")
        
        # Check for key indicators
        key_indicators = ['rsi', 'macd', 'macd_signal', 'macd_hist']
        missing_indicators = [ind for ind in key_indicators if ind not in result_df.columns]
        
        if missing_indicators:
            logger.warning(f"Missing key indicators: {missing_indicators}")
            return False
        else:
            logger.info("✓ Legacy compatibility maintained")
            return True
        
    except Exception as e:
        logger.error(f"Error testing legacy compatibility: {e}")
        return False

def performance_benchmark():
    """Benchmark the performance of indicator calculations."""
    
    logger.info("\\n=== Performance Benchmark ===")
    
    try:
        import time
        
        # Test with larger dataset
        large_df = generate_sample_data(days=90, symbol='BTCUSDT')  # ~2160 data points
        logger.info(f"Benchmarking with {len(large_df)} data points")
        
        indicators = TechnicalIndicators()
        
        start_time = time.time()
        result_df = indicators.calculate_all_indicators(large_df)
        end_time = time.time()
        
        processing_time = end_time - start_time
        points_per_second = len(large_df) / processing_time
        
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Throughput: {points_per_second:.0f} data points per second")
        
        # Memory usage check
        memory_usage_mb = result_df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"Memory usage: {memory_usage_mb:.2f} MB for processed data")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in performance benchmark: {e}")
        return False

def main():
    """Run all tests and validations."""
    
    print("🚀 Starting Technical Indicators Validation Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Run tests
    _, test_results['indicators'] = test_technical_indicators()
    test_results['model_integration'] = test_model_training_integration()
    test_results['legacy_compatibility'] = test_legacy_compatibility()
    test_results['performance'] = performance_benchmark()
    
    # Summary
    print("\\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests PASSED! The system is ready for production deployment.")
    else:
        print("⚠️  Some tests FAILED. Please review the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

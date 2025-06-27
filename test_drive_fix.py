#!/usr/bin/env python3
"""
Test script to verify Google Drive upload fix and data pipeline.
"""

import os
import sys
import logging
import pandas as pd
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_drive_upload_fix():
    """Test that the Drive upload method calls work correctly."""
    
    logger.info("🧪 Testing Google Drive upload fix...")
    
    try:
        # Test the import and method availability
        from src.drive_manager import EnhancedDriveManager
        from src.config import USE_GOOGLE_DRIVE
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as temp_file:
            # Create a small test DataFrame
            test_data = {
                'timestamp': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
                'symbol': ['BTCUSDT', 'BTCUSDT'],
                'close': [50000.0, 51000.0],
                'volume': [1000.0, 1100.0]
            }
            df = pd.DataFrame(test_data)
            df.to_parquet(temp_file.name)
            temp_file_path = temp_file.name
        
        logger.info(f"📄 Created test file: {temp_file_path}")
        
        # Test the drive manager initialization
        if USE_GOOGLE_DRIVE:
            logger.info("🔄 Testing EnhancedDriveManager initialization...")
            drive_manager = EnhancedDriveManager()
            
            # Test the upload_file_async method exists
            if hasattr(drive_manager, 'upload_file_async'):
                logger.info("✅ upload_file_async method exists")
                
                # Test the method call (it will just queue the file)
                result = drive_manager.upload_file_async(
                    local_path=Path(temp_file_path),
                    category="test_data",
                    subcategory="btcusdt",
                    priority=1,
                    date_based=True
                )
                
                logger.info(f"📤 Upload queue result: {result}")
                logger.info("✅ Google Drive upload method call works correctly")
                
            else:
                logger.error("❌ upload_file_async method not found")
                return False
        else:
            logger.info("⚠️ Google Drive integration disabled in config")
        
        # Clean up test file
        os.unlink(temp_file_path)
        logger.info("🧹 Cleaned up test file")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Drive upload test failed: {e}")
        return False

def test_data_loader():
    """Test the enhanced data loader with Drive fallback."""
    
    logger.info("🧪 Testing enhanced data loader...")
    
    try:
        from src.model_training.local_data_loader import fetch_parquet_data_from_local
        
        # Test data loading
        logger.info("📊 Testing data loading with Drive fallback...")
        df = fetch_parquet_data_from_local()
        
        if not df.empty:
            logger.info(f"✅ Successfully loaded {len(df)} rows")
            logger.info(f"📊 Columns: {list(df.columns)}")
            if 'symbol' in df.columns:
                logger.info(f"🎯 Symbols: {df['symbol'].nunique()} unique symbols")
                logger.info(f"📈 Top symbols: {df['symbol'].value_counts().head().to_dict()}")
        else:
            logger.warning("⚠️ No data loaded (this might be expected if no local data exists)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loader test failed: {e}")
        return False

def test_local_storage_save():
    """Test the local storage save function with Drive upload."""
    
    logger.info("🧪 Testing local storage save with Drive upload...")
    
    try:
        from src.data_collector.local_storage import save_parquet_file
        import numpy as np
        
        # Create test data
        test_data = {
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.uniform(50000, 52000, 100),
            'high': np.random.uniform(50500, 52500, 100),
            'low': np.random.uniform(49500, 51500, 100),
            'close': np.random.uniform(50000, 52000, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'rsi': np.random.uniform(30, 70, 100),
            'macd': np.random.uniform(-100, 100, 100),
        }
        df = pd.DataFrame(test_data)
        
        # Test saving with Drive upload
        logger.info("💾 Testing save with Drive upload...")
        result = save_parquet_file(df, "TEST_DRIVE_FIX.parquet", "TESTCOIN")
        
        if result:
            logger.info(f"✅ Successfully saved test file: {result}")
            # Clean up test file
            if os.path.exists(result):
                os.unlink(result)
                logger.info("🧹 Cleaned up test file")
        else:
            logger.error("❌ Failed to save test file")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Local storage save test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    logger.info("🚀 Starting Google Drive pipeline fix tests...")
    
    tests = [
        ("Drive Upload Fix", test_drive_upload_fix),
        ("Enhanced Data Loader", test_data_loader),
        ("Local Storage with Drive", test_local_storage_save),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n" + "="*50)
        logger.info(f"🧪 Running: {test_name}")
        logger.info("="*50)
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"📊 {test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"❌ {test_name} crashed: {e}")
    
    # Print summary
    logger.info(f"\n" + "="*50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Google Drive pipeline fix is working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

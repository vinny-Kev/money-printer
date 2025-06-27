#!/usr/bin/env python3
"""
Comprehensive test for Google Drive access and Binance balance detection.
"""

import os
import sys
import logging
import tempfile
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_google_drive_access():
    """Test Google Drive read/write access."""
    
    logger.info("🔍 Testing Google Drive Access...")
    
    try:
        from src.drive_manager import EnhancedDriveManager
        from src.config import USE_GOOGLE_DRIVE, GOOGLE_DRIVE_FOLDER_ID
        
        logger.info(f"📊 USE_GOOGLE_DRIVE setting: {USE_GOOGLE_DRIVE}")
        logger.info(f"📁 GOOGLE_DRIVE_FOLDER_ID: {GOOGLE_DRIVE_FOLDER_ID}")
        
        if not USE_GOOGLE_DRIVE:
            logger.warning("⚠️ Google Drive is disabled in config. Enable with USE_GOOGLE_DRIVE=true")
            return False
        
        # Test drive manager initialization
        drive_manager = EnhancedDriveManager()
        
        # Check authentication
        logger.info(f"🔐 Drive authenticated: {drive_manager.authenticated}")
        logger.info(f"🔄 Sync enabled: {drive_manager.sync_enabled}")
        
        if not drive_manager.authenticated:
            logger.error("❌ Google Drive not authenticated!")
            logger.error("🔑 Make sure service account key is at: Z:\\money_printer\\secrets\\service_account.json")
            return False
        
        # Test creating a test file and uploading
        logger.info("📤 Testing file upload...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Test data for Money Printer Drive validation")
            temp_file_path = temp_file.name
        
        try:
            # Test upload
            upload_result = drive_manager.upload_file_async(
                local_path=Path(temp_file_path),
                category="test",
                subcategory="validation",
                priority=1,
                date_based=False
            )
            
            logger.info(f"📤 Upload queued: {upload_result}")
            
            # Wait a moment for batch processing
            import time
            time.sleep(2)
            
            # Check if file was uploaded
            logger.info("✅ Google Drive upload test completed")
            
        finally:
            # Clean up test file
            os.unlink(temp_file_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Google Drive test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_binance_balance():
    """Test Binance balance detection."""
    
    logger.info("💰 Testing Binance Balance Detection...")
    
    try:
        from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY
        
        # Check if API keys are configured
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            logger.error("❌ Binance API keys not configured!")
            logger.error("🔑 Set BINANCE_API_KEY and BINANCE_SECRET_KEY in environment")
            return False
        
        logger.info("🔑 Binance API keys found")
        logger.info(f"📊 API Key: {BINANCE_API_KEY[:10]}...")
        
        # Test Binance client connection
        try:
            from binance.client import Client
            client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY, testnet=False)
            
            logger.info("🔄 Testing Binance connection...")
            
            # Get account info
            account_info = client.get_account()
            logger.info("✅ Binance connection successful!")
            
            # Get USDT balance
            usdt_balance = 0
            for balance in account_info['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            logger.info(f"💰 USDT Balance: {usdt_balance:.4f} USDT")
            
            if usdt_balance >= 8.0:
                logger.info("✅ Balance is sufficient for trading! ($8+ detected)")
            elif usdt_balance > 0:
                logger.warning(f"⚠️ Balance detected but low: ${usdt_balance:.2f} USD")
            else:
                logger.error("❌ No USDT balance detected!")
                logger.error("💡 Make sure you deposited USDT to your Binance Spot wallet")
            
            # Check if balance reflects your 500 PHP deposit
            php_to_usd = 500 / 58  # Approximate PHP to USD conversion
            logger.info(f"📊 Expected balance from 500 PHP: ~${php_to_usd:.2f} USD")
            
            if usdt_balance >= php_to_usd * 0.9:  # Allow 10% variance
                logger.info("✅ Balance matches expected deposit!")
            else:
                logger.warning("⚠️ Balance doesn't match expected deposit amount")
                logger.warning("💡 Check if deposit is in Spot wallet (not Futures/Margin)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Binance connection failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Balance test failed: {e}")
        return False

def test_discord_balance_command():
    """Test the Discord balance command functionality."""
    
    logger.info("🤖 Testing Discord Balance Command...")
    
    try:
        # Import the Discord bot components
        from src.lightweight_discord_bot import get_binance_balance
        
        logger.info("🔄 Testing balance retrieval function...")
        
        # Test the balance function
        balance = get_binance_balance()
        
        if balance is not None:
            logger.info(f"✅ Discord balance function works: {balance:.4f} USDT")
            return True
        else:
            logger.error("❌ Discord balance function returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Discord balance test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_data_pipeline_integration():
    """Test the complete data pipeline with Drive integration."""
    
    logger.info("📊 Testing Data Pipeline Integration...")
    
    try:
        # Test data saving with Drive upload
        from src.data_collector.local_storage import save_parquet_file
        import numpy as np
        
        # Create test data
        test_data = {
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'open': np.random.uniform(50000, 52000, 10),
            'high': np.random.uniform(50500, 52500, 10),
            'low': np.random.uniform(49500, 51500, 10),
            'close': np.random.uniform(50000, 52000, 10),
            'volume': np.random.uniform(1000, 5000, 10),
            'rsi': np.random.uniform(30, 70, 10),
            'macd': np.random.uniform(-100, 100, 10),
        }
        df = pd.DataFrame(test_data)
        
        logger.info("💾 Testing data save with Drive upload...")
        result = save_parquet_file(df, "DRIVE_TEST.parquet", "TESTCOIN")
        
        if result:
            logger.info(f"✅ Data pipeline test successful: {result}")
            # Clean up test file
            if os.path.exists(result):
                os.unlink(result)
                logger.info("🧹 Cleaned up test file")
            return True
        else:
            logger.error("❌ Data pipeline test failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Data pipeline test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    
    logger.info("🚀 Starting Comprehensive Drive & Balance Validation...")
    logger.info("="*60)
    
    tests = [
        ("Google Drive Access", test_google_drive_access),
        ("Binance Balance Detection", test_binance_balance),
        ("Discord Balance Command", test_discord_balance_command),
        ("Data Pipeline Integration", test_data_pipeline_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Running: {test_name}")
        logger.info('='*60)
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"📊 {test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"❌ {test_name} crashed: {e}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 VALIDATION RESULTS SUMMARY")
    logger.info('='*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    # Specific recommendations
    logger.info(f"\n{'='*60}")
    logger.info("💡 RECOMMENDATIONS")
    logger.info('='*60)
    
    if not results.get("Google Drive Access", False):
        logger.info("🔧 For Google Drive:")
        logger.info("   1. Add service account key to: Z:\\money_printer\\secrets\\service_account.json")
        logger.info("   2. Set USE_GOOGLE_DRIVE=true in environment")
        logger.info("   3. Verify GOOGLE_DRIVE_FOLDER_ID is correct")
    
    if not results.get("Binance Balance Detection", False):
        logger.info("🔧 For Binance Balance:")
        logger.info("   1. Verify BINANCE_API_KEY and BINANCE_SECRET_KEY are correct")
        logger.info("   2. Check that USDT is in your Spot wallet (not Futures)")
        logger.info("   3. Wait for deposit confirmation (can take 10-30 minutes)")
        logger.info("   4. Try the Discord /balance command directly")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

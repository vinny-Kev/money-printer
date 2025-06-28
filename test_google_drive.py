#!/usr/bin/env python3
"""
Google Drive Integration Test
Tests if data is being properly saved to Google Drive
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
import pandas as pd

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_google_drive_connection():
    """Test Google Drive connection and file listing"""
    logger.info("☁️ Testing Google Drive connection...")
    
    try:
        from drive_manager import EnhancedDriveManager
        
        drive_manager = EnhancedDriveManager()
        
        # Test listing files in the folder
        files = drive_manager.list_files_in_folder()  # Remove await since it's not async
        
        logger.info(f"📁 Found {len(files)} files in Google Drive folder")
        
        if files:
            logger.info("Recent files in Drive:")
            for file_info in files[:10]:  # Show first 10 files
                name = file_info.get('name', 'Unknown')
                size = file_info.get('size', '0')
                modified = file_info.get('modifiedTime', 'Unknown')
                logger.info(f"  📄 {name} - {int(size)/1024 if size.isdigit() else 0:.1f} KB - {modified}")
        else:
            logger.warning("⚠️ No files found in Google Drive folder")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Google Drive connection failed: {e}")
        return False

async def test_data_upload():
    """Test uploading sample data to Google Drive"""
    logger.info("📤 Testing data upload to Google Drive...")
    
    try:
        from drive_manager import EnhancedDriveManager
        from binance.client import Client
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Create sample data
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise Exception("Binance API credentials not configured")
        
        client = Client(api_key=api_key, api_secret=secret_key)
        
        # Get fresh data for upload test
        symbol = 'BTCUSDT'
        logger.info(f"📊 Fetching fresh data for {symbol}...")
        
        klines = client.get_klines(symbol=symbol, interval='1h', limit=50)
        
        if not klines:
            raise Exception("No data retrieved from Binance")
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to readable format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert price and volume columns to float
        price_volume_cols = ['open', 'high', 'low', 'close', 'volume', 
                           'quote_asset_volume', 'taker_buy_base_asset_volume', 
                           'taker_buy_quote_asset_volume']
        for col in price_volume_cols:
            df[col] = df[col].astype(float)
        
        # Save locally first
        local_dir = "data/scraped_data/parquet_files"
        os.makedirs(local_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_filename = f"{local_dir}/{symbol}_test_upload_{timestamp}.parquet"
        df.to_parquet(local_filename, index=False)
        
        logger.info(f"💾 Saved {len(df)} rows locally: {local_filename}")
        
        # Upload to Google Drive
        drive_manager = EnhancedDriveManager()
        
        remote_filename = f"{symbol}_test_upload_{timestamp}.parquet"
        
        result = await drive_manager.upload_file_async(local_filename, remote_filename)
        
        if result:
            logger.info(f"☁️ Successfully uploaded to Google Drive: {remote_filename}")
            logger.info(f"📁 File ID: {result}")
            return True
        else:
            logger.error("❌ Failed to upload to Google Drive")
            return False
        
    except Exception as e:
        logger.error(f"❌ Data upload test failed: {e}")
        return False

async def test_local_to_drive_sync():
    """Test syncing local data files to Google Drive"""
    logger.info("🔄 Testing local to Drive sync...")
    
    try:
        from data_collector.local_storage import save_parquet_file
        from drive_manager import EnhancedDriveManager
        
        # Check for existing local files
        local_dir = "data/scraped_data/parquet_files"
        
        if not os.path.exists(local_dir):
            logger.warning("⚠️ No local data directory found")
            return False
        
        parquet_files = [f for f in os.listdir(local_dir) if f.endswith('.parquet')]
        
        if not parquet_files:
            logger.warning("⚠️ No parquet files found locally")
            return False
        
        logger.info(f"📊 Found {len(parquet_files)} local parquet files")
        
        # Test uploading one recent file
        recent_file = sorted(parquet_files)[-1]  # Get most recent file
        file_path = os.path.join(local_dir, recent_file)
        
        logger.info(f"📤 Uploading recent file: {recent_file}")
        
        drive_manager = EnhancedDriveManager()
        
        result = await drive_manager.upload_file_async(file_path, f"sync_test_{recent_file}")
        
        if result:
            logger.info(f"✅ Successfully synced file to Drive")
            return True
        else:
            logger.error("❌ Failed to sync file to Drive")
            return False
        
    except Exception as e:
        logger.error(f"❌ Local to Drive sync test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🏁 Starting Google Drive Integration Test")
    logger.info("=" * 60)
    
    tests = [
        ("Google Drive Connection", test_google_drive_connection),
        ("Data Upload Test", test_data_upload),
        ("Local to Drive Sync", test_local_to_drive_sync),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! Google Drive integration is working.")
        logger.info("💡 Data should now be automatically saved to Google Drive during collection.")
    else:
        logger.warning("⚠️ Some tests failed. Check Google Drive configuration:")
        logger.warning("   1. Ensure service account key is in secrets/service_account.json")
        logger.warning("   2. Check GOOGLE_DRIVE_FOLDER_ID in .env file")
        logger.warning("   3. Verify Google Drive API is enabled")
    
    return passed == len(tests)

if __name__ == "__main__":
    asyncio.run(main())

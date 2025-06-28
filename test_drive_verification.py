#!/usr/bin/env python3
"""
Google Drive Verification Test
Checks if files are actually being saved to Google Drive (Railway deployment)
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_drive_connectivity():
    """Test basic Google Drive connectivity"""
    logger.info("🔗 Testing Google Drive connectivity...")
    
    try:
        from drive_manager import EnhancedDriveManager
        
        drive_manager = EnhancedDriveManager()
        
        # List files in the Drive folder
        files = drive_manager.list_files_in_folder()
        
        logger.info(f"📁 Found {len(files)} files in Google Drive folder")
        
        if files:
            logger.info("📋 Recent files in Drive:")
            for i, file_info in enumerate(files[:10]):  # Show first 10 files
                name = file_info.get('name', 'Unknown')
                size = file_info.get('size', '0')
                modified = file_info.get('modifiedTime', 'Unknown')
                file_size_kb = int(size) / 1024 if size.isdigit() else 0
                logger.info(f"  {i+1:2d}. 📄 {name} - {file_size_kb:.1f} KB - {modified}")
        else:
            logger.warning("⚠️ No files found in Google Drive folder")
        
        return len(files) > 0
        
    except Exception as e:
        logger.error(f"❌ Google Drive connectivity test failed: {e}")
        return False

async def create_and_upload_test_file():
    """Create a test file and upload it to Google Drive"""
    logger.info("📤 Creating and uploading test file...")
    
    try:
        from drive_manager import EnhancedDriveManager
        from binance.client import Client
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Get sample data from Binance
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise Exception("Binance API credentials not configured")
        
        client = Client(api_key=api_key, api_secret=secret_key)
        
        # Get fresh data
        symbol = 'BTCUSDT'
        logger.info(f"📊 Fetching fresh data for {symbol}...")
        
        klines = client.get_klines(symbol=symbol, interval='1h', limit=24)  # Last 24 hours
        
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
        price_volume_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_volume_cols:
            df[col] = df[col].astype(float)
        
        # Create a temporary file in memory
        import tempfile
        import io
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        # Create temporary file for upload
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            temp_file.write(buffer.getvalue())
            temp_file_path = temp_file.name
        
        logger.info(f"💾 Created temporary file with {len(df)} rows")
        
        # Upload to Google Drive
        drive_manager = EnhancedDriveManager()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        remote_filename = f"railway_test_{symbol}_{timestamp}.parquet"
        
        logger.info(f"☁️ Uploading to Drive as: {remote_filename}")
        
        result = await drive_manager.upload_file_async(temp_file_path, remote_filename)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        if result:
            logger.info(f"✅ Successfully uploaded test file to Google Drive")
            logger.info(f"📁 File ID: {result}")
            logger.info(f"📊 Data summary:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Rows: {len(df)}")
            logger.info(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
            logger.info(f"   Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            return True
        else:
            logger.error("❌ Failed to upload test file to Google Drive")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test file upload failed: {e}")
        return False

async def verify_file_exists_in_drive():
    """Verify that our uploaded file actually exists in Drive"""
    logger.info("🔍 Verifying uploaded files exist in Drive...")
    
    try:
        from drive_manager import EnhancedDriveManager
        
        drive_manager = EnhancedDriveManager()
        
        # Get fresh file list
        files = drive_manager.list_files_in_folder()
        
        # Look for our test files
        test_files = [f for f in files if 'railway_test_' in f.get('name', '')]
        
        logger.info(f"🎯 Found {len(test_files)} test files in Drive")
        
        if test_files:
            logger.info("📋 Test files found:")
            for file_info in test_files:
                name = file_info.get('name', 'Unknown')
                size = file_info.get('size', '0')
                modified = file_info.get('modifiedTime', 'Unknown')
                file_size_kb = int(size) / 1024 if size.isdigit() else 0
                logger.info(f"  📄 {name} - {file_size_kb:.1f} KB - {modified}")
            return True
        else:
            logger.warning("⚠️ No test files found in Drive")
            return False
        
    except Exception as e:
        logger.error(f"❌ File verification failed: {e}")
        return False

async def check_drive_permissions():
    """Check if we have proper permissions to read/write to Drive"""
    logger.info("🔐 Checking Google Drive permissions...")
    
    try:
        from drive_manager import EnhancedDriveManager
        
        drive_manager = EnhancedDriveManager()
        
        # Try to get folder info
        folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        if not folder_id:
            logger.error("❌ GOOGLE_DRIVE_FOLDER_ID not configured")
            return False
        
        logger.info(f"📁 Using folder ID: {folder_id}")
        
        # Check if service account key exists
        service_key_path = "secrets/service_account.json"
        if os.path.exists(service_key_path):
            logger.info("✅ Service account key file found")
        else:
            logger.error(f"❌ Service account key missing: {service_key_path}")
            return False
        
        # Try listing files (this tests read permissions)
        files = drive_manager.list_files_in_folder()
        logger.info(f"✅ Read permissions OK - can list {len(files)} files")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Permission check failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🏁 Starting Google Drive Verification Test")
    logger.info("🚀 Railway Deployment - Drive-Only Storage Test")
    logger.info("=" * 60)
    
    tests = [
        ("Drive Permissions", check_drive_permissions),
        ("Drive Connectivity", test_drive_connectivity),
        ("Upload Test File", create_and_upload_test_file),
        ("Verify File Exists", verify_file_exists_in_drive),
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
    logger.info("📊 Google Drive Test Results:")
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! Google Drive integration is working perfectly.")
        logger.info("💡 Files will be saved directly to Google Drive during data collection.")
    else:
        logger.warning("⚠️ Some tests failed. Data may not be saved properly:")
        if not results.get("Drive Permissions", False):
            logger.warning("   🔐 Check service account key and folder permissions")
        if not results.get("Drive Connectivity", False):
            logger.warning("   🔗 Check network connectivity and API access")
        if not results.get("Upload Test File", False):
            logger.warning("   📤 Check upload functionality and quotas")
    
    return passed == len(tests)

if __name__ == "__main__":
    asyncio.run(main())

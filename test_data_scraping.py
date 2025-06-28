#!/usr/bin/env python3
"""
Quick Data Scraper Test
Tests the data scraping functionality and creates sample data for model training
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

def test_binance_connection():
    """Test basic Binance connection and data retrieval"""
    logger.info("🔧 Testing Binance connection...")
    
    try:
        from binance.client import Client
        from dotenv import load_dotenv
        
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise Exception("Binance API credentials not configured")
        
        client = Client(api_key=api_key, api_secret=secret_key)
        
        # Test connection
        client.ping()
        logger.info("✅ Binance connection successful")
        
        # Test data retrieval
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        for symbol in symbols:
            logger.info(f"📊 Testing data retrieval for {symbol}...")
            
            # Get recent klines (1 hour interval, last 100 candles)
            klines = client.get_klines(symbol=symbol, interval='1h', limit=100)
            
            if klines:
                logger.info(f"✅ Retrieved {len(klines)} klines for {symbol}")
                
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
                
                # Save to local storage
                data_dir = "data/scraped_data/parquet_files"
                os.makedirs(data_dir, exist_ok=True)
                
                filename = f"{data_dir}/{symbol}_1h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.to_parquet(filename, index=False)
                
                logger.info(f"💾 Saved {len(df)} rows to {filename}")
                
                # Show sample data
                logger.info(f"📈 Sample data for {symbol}:")
                logger.info(f"   Latest close price: ${df['close'].iloc[-1]:.2f}")
                logger.info(f"   Latest volume: {df['volume'].iloc[-1]:.2f}")
                logger.info(f"   Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
                
            else:
                logger.warning(f"⚠️ No data retrieved for {symbol}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Binance connection test failed: {e}")
        return False

def test_data_storage():
    """Test local data storage functionality"""
    logger.info("💾 Testing data storage functionality...")
    
    try:
        # Check if data directory exists
        data_dir = "data/scraped_data/parquet_files"
        
        if not os.path.exists(data_dir):
            logger.warning(f"⚠️ Data directory doesn't exist: {data_dir}")
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"✅ Created data directory: {data_dir}")
        
        # List existing files
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        logger.info(f"📊 Found {len(parquet_files)} existing parquet files")
        
        if parquet_files:
            logger.info("Recent files:")
            for file in sorted(parquet_files)[-5:]:  # Show last 5 files
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                logger.info(f"  📄 {file} - {file_size:.1f} KB - {file_time}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data storage test failed: {e}")
        return False

def test_discord_notifications():
    """Test Discord notification system"""
    logger.info("🤖 Testing Discord notifications...")
    
    try:
        from discord_notifications import send_scraper_notification
        
        test_message = f"🧪 Data scraper test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        result = send_scraper_notification(test_message)
        
        if result:
            logger.info("✅ Discord notification sent successfully")
        else:
            logger.warning("⚠️ Discord notification may have failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Discord notification test failed: {e}")
        return False

def run_quick_data_collection():
    """Run a quick data collection for testing purposes"""
    logger.info("🚀 Running quick data collection...")
    
    try:
        success = test_binance_connection()
        if not success:
            logger.error("❌ Binance connection failed - cannot proceed with data collection")
            return False
        
        logger.info("✅ Quick data collection completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick data collection failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🏁 Starting Data Scraper Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Binance Connection", test_binance_connection),
        ("Data Storage", test_data_storage),
        ("Discord Notifications", test_discord_notifications),
        ("Quick Data Collection", run_quick_data_collection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
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
        logger.info("🎉 All tests passed! Data scraping functionality is working.")
    else:
        logger.warning("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()

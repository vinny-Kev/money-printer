#!/usr/bin/env python3
"""
Simple Data Collection Script with Proper Signal Handling
Collects data for a limited time and saves to both local storage and Google Drive
"""

import os
import sys
import signal
import logging
import time
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

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global running
    logger.info(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
    running = False

def collect_sample_data():
    """Collect sample data from Binance and save it"""
    logger.info("📊 Starting sample data collection...")
    
    try:
        from binance.client import Client
        from dotenv import load_dotenv
        from data_collector.local_storage import save_parquet_file
        
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise Exception("Binance API credentials not configured")
        
        client = Client(api_key=api_key, api_secret=secret_key)
        
        # Symbols to collect data for
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT']
        
        collected_count = 0
        
        for i, symbol in enumerate(symbols):
            if not running:
                logger.info("🛑 Stopping data collection due to signal")
                break
                
            logger.info(f"📈 Collecting data for {symbol} ({i+1}/{len(symbols)})...")
            
            try:
                # Get recent klines (1 hour interval, last 100 candles)
                klines = client.get_klines(symbol=symbol, interval='1h', limit=100)
                
                if klines:
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
                    
                    # Save using the local storage function (which also handles Drive upload)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{symbol}_1h_{timestamp}.parquet"
                    
                    result = save_parquet_file(df, filename, symbol)
                    
                    if result:
                        logger.info(f"✅ Successfully saved {len(df)} rows for {symbol}")
                        logger.info(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
                        logger.info(f"   Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
                        collected_count += 1
                    else:
                        logger.error(f"❌ Failed to save data for {symbol}")
                        
                else:
                    logger.warning(f"⚠️ No data retrieved for {symbol}")
                
                # Small delay between requests to be nice to the API
                if running:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ Error collecting data for {symbol}: {e}")
        
        logger.info(f"📊 Data collection completed. Successfully collected data for {collected_count}/{len(symbols)} symbols.")
        return collected_count
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        return 0

def main():
    """Main function with signal handling"""
    global running
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting Simple Data Collection")
    logger.info("💡 Press Ctrl+C to stop gracefully")
    logger.info("=" * 60)
    
    try:
        # Check configuration first
        from dotenv import load_dotenv
        load_dotenv()
        
        use_drive = os.getenv('USE_GOOGLE_DRIVE', 'false').lower() == 'true'
        logger.info(f"☁️ Google Drive integration: {'✅ ENABLED' if use_drive else '❌ DISABLED'}")
        
        if use_drive:
            # Check if service account key exists
            service_key_path = "secrets/service_account.json"
            if os.path.exists(service_key_path):
                logger.info("🔑 Google Drive service account key found")
            else:
                logger.warning("⚠️ Google Drive service account key missing - uploads will fail")
                logger.warning(f"   Expected location: {service_key_path}")
        
        logger.info("\n🎯 Starting data collection...")
        
        # Collect data
        collected = collect_sample_data()
        
        if collected > 0:
            logger.info(f"\n✅ Data collection successful! Collected data for {collected} symbols.")
            logger.info("💾 Data saved locally and uploaded to Google Drive (if configured)")
        else:
            logger.warning("\n⚠️ No data was collected. Check your configuration.")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Data collection interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Data collection failed: {e}")
    finally:
        logger.info("🏁 Data collection finished")

if __name__ == "__main__":
    main()

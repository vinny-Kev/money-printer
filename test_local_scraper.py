#!/usr/bin/env python3
"""
Local Test Scraper - Minimal version to test if data collection works
This will test:
1. Binance API connection
2. WebSocket data reception
3. Local file saving
4. Google Drive upload (if enabled)
"""
import os
import sys
import time
import json
import pandas as pd
import logging
from datetime import datetime
from binance import ThreadedWebsocketManager, Client

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, USE_GOOGLE_DRIVE
from src.data_collector.local_storage import save_parquet_file
from src.discord_notifications import send_scraper_notification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Test data buffer
test_buffer = {}
message_count = 0
start_time = time.time()

def test_kline_handler(msg):
    """Test kline message handler"""
    global message_count, test_buffer
    
    if msg.get('e') != 'kline':
        return
    
    message_count += 1
    symbol = msg['s']
    k = msg['k']
    
    # Log every message for debugging
    logger.info(f"📈 Message #{message_count}: {symbol} @ ${float(k['c']):.4f}")
    
    # Convert to our format
    kline_data = {
        'timestamp': pd.to_datetime(k['t'], unit='ms'),
        'open': float(k['o']),
        'high': float(k['h']),
        'low': float(k['l']),
        'close': float(k['c']),
        'volume': float(k['v'])
    }
    
    # Add to buffer
    if symbol not in test_buffer:
        test_buffer[symbol] = []
    
    test_buffer[symbol].append(kline_data)
    
    # Log buffer status
    buffer_size = len(test_buffer[symbol])
    logger.info(f"📊 {symbol} buffer: {buffer_size} records")
    
    # Save after 5 messages per symbol
    if buffer_size >= 5:
        logger.info(f"💾 Saving {symbol} data (5 records reached)")
        save_test_data(symbol)

def save_test_data(symbol):
    """Save test data to file"""
    try:
        if symbol not in test_buffer or not test_buffer[symbol]:
            logger.warning(f"No data to save for {symbol}")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(test_buffer[symbol])
        
        # Save using our local storage function
        filename = f"{symbol}.parquet"
        result = save_parquet_file(df, filename, symbol)
        
        if result:
            file_size = os.path.getsize(result)
            logger.info(f"✅ Successfully saved {symbol}: {len(df)} records, {file_size} bytes")
            logger.info(f"📁 File location: {result}")
            
            # Test Google Drive if enabled
            if USE_GOOGLE_DRIVE:
                logger.info(f"☁️ Google Drive upload should have been triggered")
            else:
                logger.info(f"☁️ Google Drive disabled (USE_GOOGLE_DRIVE=False)")
            
        else:
            logger.error(f"❌ Failed to save {symbol}")
        
        # Clear buffer for this symbol
        test_buffer[symbol].clear()
        logger.info(f"🗑️ Cleared buffer for {symbol}")
        
    except Exception as e:
        logger.error(f"❌ Error saving {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main test function"""
    logger.info("🧪 Starting Local Test Scraper")
    logger.info(f"⏰ Start time: {datetime.now()}")
    
    # Test API credentials
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        logger.error("❌ Missing Binance API credentials")
        return
    
    logger.info("🔑 API credentials found")
    
    # Test Binance client
    try:
        client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
        server_time = client.get_server_time()
        logger.info(f"✅ Binance connection successful. Server time: {server_time}")
    except Exception as e:
        logger.error(f"❌ Binance connection failed: {e}")
        return
    
    # Test symbols (just 3 for quick testing)
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    logger.info(f"🎯 Testing with symbols: {test_symbols}")
    
    # Initialize WebSocket manager
    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
    
    try:
        # Start WebSocket manager
        twm.start()
        logger.info("🚀 WebSocket manager started")
        
        # Subscribe to test symbols
        for symbol in test_symbols:
            twm.start_kline_socket(callback=test_kline_handler, symbol=symbol.lower(), interval="1m")
            logger.info(f"📡 Subscribed to {symbol}")
            time.sleep(0.1)  # Small delay between subscriptions
        
        # Send Discord notification
        try:
            send_scraper_notification("🧪 **Test Scraper Started**: Local testing mode with 3 symbols")
        except Exception as e:
            logger.warning(f"Discord notification failed: {e}")
        
        # Run for 2 minutes
        test_duration = 120  # 2 minutes
        logger.info(f"⏱️ Running test for {test_duration} seconds...")
        
        start = time.time()
        while (time.time() - start) < test_duration:
            elapsed = int(time.time() - start)
            if elapsed % 15 == 0:  # Log every 15 seconds
                total_messages = sum(len(data) for data in test_buffer.values())
                logger.info(f"📊 Status: {elapsed}s elapsed, {message_count} messages, {total_messages} buffered")
            time.sleep(1)
        
        logger.info("⏰ Test duration complete, saving remaining data...")
        
        # Save any remaining data
        for symbol in test_symbols:
            if symbol in test_buffer and test_buffer[symbol]:
                save_test_data(symbol)
        
        # Final summary
        runtime = time.time() - start_time
        logger.info(f"📊 Test Summary:")
        logger.info(f"   Runtime: {runtime:.1f} seconds")
        logger.info(f"   Messages received: {message_count}")
        logger.info(f"   Symbols tested: {len(test_symbols)}")
        
        # Send final notification
        try:
            summary = (
                f"🧪 **Test Scraper Complete**\n"
                f"⏱️ Runtime: {runtime:.1f}s\n"
                f"📨 Messages: {message_count}\n"
                f"💾 Check local files and logs!"
            )
            send_scraper_notification(summary)
        except Exception as e:
            logger.warning(f"Final Discord notification failed: {e}")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        try:
            twm.stop()
            logger.info("🛑 WebSocket manager stopped")
        except:
            pass

if __name__ == "__main__":
    main()

"""
Railway-Optimized Data Scraper
Saves data directly to Google Drive without local storage
"""
import os
import logging
import asyncio
import signal
import sys
import time
import threading
from datetime import datetime, timezone, timedelta
import pandas as pd
from binance import Client
import tempfile
import io

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global variables
ohlcv_buffer = {}
stop_collection = threading.Event()

class RailwayDataScraper:
    def __init__(self):
        self.client = None
        self.drive_manager = None
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT']
        self.collection_interval = 300  # 5 minutes
        self.files_uploaded = 0
        
    async def initialize(self):
        """Initialize Binance client and Drive manager"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Initialize Binance client
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            
            if not api_key or not secret_key:
                raise Exception("Binance API credentials not configured")
            
            self.client = Client(api_key=api_key, api_secret=secret_key)
            self.client.ping()  # Test connection
            logger.info("✅ Binance client initialized")
            
            # Initialize Drive manager
            import sys
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from src.drive_manager import EnhancedDriveManager
            self.drive_manager = EnhancedDriveManager()
            logger.info("✅ Google Drive manager initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def get_market_data(self, symbol, interval='1h', limit=100):
        """Get market data from Binance"""
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            if not klines:
                logger.warning(f"⚠️ No data received for {symbol}")
                return None
            
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
            
            logger.info(f"📊 Retrieved {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to get data for {symbol}: {e}")
            return None
    
    async def save_to_drive(self, df, symbol):
        """Save DataFrame directly to Google Drive"""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_1h_{timestamp}.parquet"
            
            # Create temporary file in memory
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            
            # Create temporary file for upload
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                temp_file.write(buffer.getvalue())
                temp_file_path = temp_file.name
            
            # Convert to Path object
            from pathlib import Path
            temp_path = Path(temp_file_path)
            
            # Rename the temp file to have the proper filename
            final_temp_path = temp_path.parent / filename
            temp_path.rename(final_temp_path)
            
            # Upload to Google Drive using the batch system
            result = self.drive_manager.upload_file_async(
                final_temp_path, 
                "market_data", 
                "scraped", 
                priority=1, 
                date_based=True
            )
            
            # Clean up temporary file
            final_temp_path.unlink()
            
            if result:
                self.files_uploaded += 1
                logger.info(f"☁️ Successfully queued {filename} for Google Drive upload")
                return True
            else:
                logger.error(f"❌ Failed to queue {filename} for Google Drive upload")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error saving {symbol} to Drive: {e}")
            return False
    
    async def collect_and_save_data(self):
        """Collect data for all symbols and save to Drive"""
        logger.info(f"🔄 Starting data collection cycle for {len(self.symbols)} symbols...")
        
        successful_uploads = 0
        
        for symbol in self.symbols:
            if stop_collection.is_set():
                logger.info("🛑 Collection stopped during symbol processing")
                break
                
            try:
                # Get market data
                df = self.get_market_data(symbol)
                
                if df is not None and len(df) > 0:
                    # Save directly to Google Drive
                    if await self.save_to_drive(df, symbol):
                        successful_uploads += 1
                        logger.info(f"✅ {symbol}: {len(df)} records saved to Drive")
                        logger.info(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
                    else:
                        logger.error(f"❌ {symbol}: Failed to save to Drive")
                else:
                    logger.warning(f"⚠️ {symbol}: No data to save")
                
                # Small delay between symbols
                if not stop_collection.is_set():
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
        
        logger.info(f"📊 Collection cycle completed: {successful_uploads}/{len(self.symbols)} files saved")
        return successful_uploads
    
    async def run_collection(self, stop_flag=None):
        """Main collection loop"""
        global stop_collection
        
        if stop_flag:
            stop_collection = stop_flag
        
        logger.info("🚀 Starting Railway Data Collection (Drive-Only)")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"⏱️ Collection interval: {self.collection_interval} seconds")
        
        # Initialize connections
        if not await self.initialize():
            logger.error("❌ Failed to initialize. Exiting.")
            return
        
        cycle_count = 0
        start_time = datetime.now()
        
        try:
            while not stop_collection.is_set():
                cycle_count += 1
                cycle_start = datetime.now()
                
                logger.info(f"\n{'='*20} Cycle {cycle_count} {'='*20}")
                logger.info(f"🕒 Started at: {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Collect and save data
                uploaded = await self.collect_and_save_data()
                
                cycle_end = datetime.now()
                cycle_duration = (cycle_end - cycle_start).total_seconds()
                
                logger.info(f"⏱️ Cycle {cycle_count} completed in {cycle_duration:.1f}s")
                logger.info(f"📊 Total files uploaded: {self.files_uploaded}")
                
                # Wait for next cycle or until stop signal
                remaining_time = self.collection_interval - cycle_duration
                if remaining_time > 0 and not stop_collection.is_set():
                    logger.info(f"😴 Waiting {remaining_time:.1f}s until next cycle...")
                    
                    # Wait in small increments to check stop signal
                    wait_time = 0
                    while wait_time < remaining_time and not stop_collection.is_set():
                        await asyncio.sleep(min(10, remaining_time - wait_time))
                        wait_time += 10
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Collection interrupted by user")
        except Exception as e:
            logger.error(f"❌ Collection error: {e}")
        finally:
            end_time = datetime.now()
            total_duration = end_time - start_time
            
            logger.info(f"\n📊 Collection Summary:")
            logger.info(f"   Duration: {total_duration}")
            logger.info(f"   Cycles completed: {cycle_count}")
            logger.info(f"   Files uploaded: {self.files_uploaded}")
            logger.info(f"   Average per cycle: {self.files_uploaded/cycle_count if cycle_count > 0 else 0:.1f}")

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"\n🛑 Received signal {signum}. Stopping collection...")
        stop_collection.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main(stop_flag=None):
    """Main function"""
    setup_signal_handlers()
    
    scraper = RailwayDataScraper()
    await scraper.run_collection(stop_flag)

if __name__ == "__main__":
    asyncio.run(main())

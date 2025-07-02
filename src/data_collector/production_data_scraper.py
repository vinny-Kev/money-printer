"""
Production Data Scraper - Robust, fallback-enabled data collection
Collects real market data with multiple storage options and graceful shutdown
"""
import asyncio
import signal
import sys
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import threading
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from binance.client import Client as BinanceClient
from src.storage.enhanced_storage_manager import EnhancedStorageManager
from src.binance_wrapper import EnhancedBinanceClient
from src.config import Config

logger = logging.getLogger(__name__)

class ProductionDataScraper:
    """Production-ready data scraper with robust error handling"""
    
    def __init__(self, 
                 symbols: List[str] = None,
                 intervals: List[str] = None,
                 drive_folder_id: str = None,
                 local_backup_dir: str = None,
                 memory_only: bool = False):
        
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        self.intervals = intervals or ['1m', '5m', '15m', '1h']
        self.stop_flag = threading.Event()
        self.is_running = False
        
        # Initialize storage manager
        self.storage = EnhancedStorageManager(
            drive_folder_id=drive_folder_id,
            local_backup_dir=local_backup_dir,
            memory_only=memory_only
        )
        
        # Initialize Binance client
        try:
            from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_TESTNET
            self.binance = EnhancedBinanceClient(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_SECRET_KEY,
                testnet=BINANCE_TESTNET
            )
            if self.binance.test_connection():
                logger.info("✅ Enhanced Binance client initialized successfully")
            else:
                logger.error("❌ Binance connection test failed")
                self.binance = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Binance client: {e}")
            self.binance = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_files_saved': 0,
            'total_rows_collected': 0,
            'errors': [],
            'successful_uploads': 0,
            'failed_uploads': 0
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.stop_flag.set()
    
    def collect_symbol_data(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Collect data for a specific symbol and interval"""
        if not self.binance:
            logger.error("❌ Binance client not available")
            return None
        
        try:
            # Get historical klines using Enhanced Binance client
            klines = self.binance.get_historical_klines(symbol, interval, limit)
            if not klines:
                logger.warning(f"⚠️ No data returned for {symbol} {interval}")
                return None
            
            # Convert to DataFrame using the wrapper method
            df = self.binance.klines_to_dataframe(klines, symbol, interval)
            
            logger.info(f"📊 Collected {len(df)} rows for {symbol} {interval}")
            return df
            
        except Exception as e:
            error_msg = f"Data collection failed for {symbol} {interval}: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return None
    
    def save_data_batch(self, data_batch: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Save a batch of collected data"""
        batch_results = {
            'timestamp': datetime.now().isoformat(),
            'files_processed': 0,
            'total_rows': 0,
            'successful_saves': 0,
            'failed_saves': 0,
            'details': []
        }
        
        for key, df in data_batch.items():
            if df is None or len(df) == 0:
                continue
            
            try:
                # Generate filename
                symbol, interval = key.split('_')
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"production_{symbol}_{interval}_{timestamp_str}.parquet"
                
                # Save data
                save_result = self.storage.save_data(df, filename)
                
                # PRODUCTION FIX: Comprehensive audit logging with absolute paths
                logger.info(f"📁 SAVE AUDIT: {filename}")
                logger.info(f"   📊 Rows: {len(df)}")
                logger.info(f"   💾 Memory: {'✅' if save_result['memory_success'] else '❌'}")
                logger.info(f"   🏠 Local: {'✅' if save_result['local_success'] else '❌'}")
                logger.info(f"   ☁️ Drive: {'✅' if save_result['drive_success'] else '❌'}")
                
                if save_result.get('errors'):
                    logger.error(f"   ❌ Errors: {save_result['errors']}")
                
                # Log all file paths for complete audit trail
                storage_status = self.storage.get_storage_status()
                logger.info(f"   📈 Total files in storage: Memory={storage_status['memory_files']}, Local={storage_status['local_files']}, Drive={storage_status['drive_files']}")
                
                batch_results['files_processed'] += 1
                batch_results['total_rows'] += len(df)
                
                if save_result['drive_success'] or save_result['local_success'] or save_result['memory_success']:
                    batch_results['successful_saves'] += 1
                    self.stats['successful_uploads'] += 1
                    logger.info(f"✅ Saved {filename} ({len(df)} rows)")
                else:
                    batch_results['failed_saves'] += 1
                    self.stats['failed_uploads'] += 1
                    logger.error(f"❌ Failed to save {filename}")
                
                batch_results['details'].append(save_result)
                
            except Exception as e:
                error_msg = f"Batch save failed for {key}: {e}"
                logger.error(f"❌ {error_msg}")
                batch_results['failed_saves'] += 1
                self.stats['errors'].append(error_msg)
        
        return batch_results
    
    def run_collection_cycle(self) -> Dict[str, Any]:
        """Run one complete data collection cycle"""
        cycle_start = time.time()
        logger.info("🔄 Starting data collection cycle...")
        
        # Collect data for all symbol/interval combinations
        data_batch = {}
        
        for symbol in self.symbols:
            if self.stop_flag.is_set():
                logger.info("🛑 Stop flag set during collection, breaking...")
                break
                
            for interval in self.intervals:
                if self.stop_flag.is_set():
                    break
                
                key = f"{symbol}_{interval}"
                logger.debug(f"📈 Collecting {key}...")
                
                data = self.collect_symbol_data(symbol, interval)
                if data is not None:
                    data_batch[key] = data
                    self.stats['total_rows_collected'] += len(data)
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
        
        # Save collected data
        batch_results = self.save_data_batch(data_batch)
        self.stats['total_files_saved'] += batch_results['successful_saves']
        
        cycle_time = time.time() - cycle_start
        logger.info(f"✅ Collection cycle completed in {cycle_time:.1f}s - "
                   f"Files: {batch_results['successful_saves']}/{batch_results['files_processed']}, "
                   f"Rows: {batch_results['total_rows']}")
        
        return batch_results
    
    def run_timed_collection(self, hours: float, cycle_interval_minutes: int = 5) -> Dict[str, Any]:
        """Run data collection for a specified number of hours"""
        if hours <= 0:
            logger.error("❌ Collection time must be positive")
            return {'error': 'Invalid collection time'}
        
        self.stats['start_time'] = datetime.now()
        self.is_running = True
        end_time = self.stats['start_time'] + timedelta(hours=hours)
        
        logger.info(f"🚀 Starting timed data collection for {hours} hours")
        logger.info(f"📅 Collection will end at: {end_time}")
        logger.info(f"🔄 Cycle interval: {cycle_interval_minutes} minutes")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"⏱️ Intervals: {', '.join(self.intervals)}")
        
        cycle_count = 0
        next_cycle_time = time.time()
        
        try:
            while datetime.now() < end_time and not self.stop_flag.is_set():
                current_time = time.time()
                
                # Check if it's time for the next cycle
                if current_time >= next_cycle_time:
                    cycle_count += 1
                    remaining_hours = (end_time - datetime.now()).total_seconds() / 3600
                    
                    logger.info(f"📊 Cycle {cycle_count} - {remaining_hours:.1f}h remaining")
                    
                    # Run collection cycle
                    cycle_results = self.run_collection_cycle()
                    
                    # Schedule next cycle
                    next_cycle_time = current_time + (cycle_interval_minutes * 60)
                    
                    # Log progress
                    self._log_progress()
                
                # Sleep briefly to avoid busy waiting
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("🛑 Collection interrupted by user")
        except Exception as e:
            logger.error(f"❌ Collection error: {e}")
            self.stats['errors'].append(f"Collection error: {e}")
        finally:
            self.is_running = False
            self.stats['end_time'] = datetime.now()
            
            # Final progress report
            self._log_final_summary()
            
            return self.get_collection_summary()
    
    def run_continuous_collection(self, cycle_interval_minutes: int = 5) -> Dict[str, Any]:
        """Run continuous data collection until stopped"""
        self.stats['start_time'] = datetime.now()
        self.is_running = True
        
        logger.info("🚀 Starting continuous data collection")
        logger.info("🛑 Press Ctrl+C to stop gracefully")
        logger.info(f"🔄 Cycle interval: {cycle_interval_minutes} minutes")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"⏱️ Intervals: {', '.join(self.intervals)}")
        
        cycle_count = 0
        next_cycle_time = time.time()
        
        try:
            while not self.stop_flag.is_set():
                current_time = time.time()
                
                # Check if it's time for the next cycle
                if current_time >= next_cycle_time:
                    cycle_count += 1
                    
                    logger.info(f"📊 Cycle {cycle_count} - Running since {self.stats['start_time']}")
                    
                    # Run collection cycle
                    cycle_results = self.run_collection_cycle()
                    
                    # Schedule next cycle
                    next_cycle_time = current_time + (cycle_interval_minutes * 60)
                    
                    # Log progress every 10 cycles
                    if cycle_count % 10 == 0:
                        self._log_progress()
                
                # Sleep briefly to avoid busy waiting
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("🛑 Collection interrupted by user")
        except Exception as e:
            logger.error(f"❌ Collection error: {e}")
            self.stats['errors'].append(f"Collection error: {e}")
        finally:
            self.is_running = False
            self.stats['end_time'] = datetime.now()
            
            # Final progress report
            self._log_final_summary()
            
            return self.get_collection_summary()
    
    def _log_progress(self):
        """Log current progress"""
        runtime = datetime.now() - self.stats['start_time']
        runtime_hours = runtime.total_seconds() / 3600
        
        storage_status = self.storage.get_storage_status()
        
        logger.info(f"📈 Progress Report:")
        logger.info(f"   ⏱️ Runtime: {runtime_hours:.1f} hours")
        logger.info(f"   📁 Files saved: {self.stats['total_files_saved']}")
        logger.info(f"   📊 Rows collected: {self.stats['total_rows_collected']:,}")
        logger.info(f"   ✅ Successful uploads: {self.stats['successful_uploads']}")
        logger.info(f"   ❌ Failed uploads: {self.stats['failed_uploads']}")
        logger.info(f"   💾 Memory files: {storage_status['memory_files']}")
        logger.info(f"   🏠 Local files: {storage_status['local_files']}")
        logger.info(f"   ☁️ Drive files: {storage_status['drive_files']}")
        
        if self.stats['errors']:
            logger.warning(f"   ⚠️ Errors: {len(self.stats['errors'])}")
    
    def _log_final_summary(self):
        """Log final collection summary"""
        if not self.stats['start_time']:
            return
        
        runtime = self.stats['end_time'] - self.stats['start_time']
        runtime_hours = runtime.total_seconds() / 3600
        
        logger.info("=" * 60)
        logger.info("📊 FINAL COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"⏱️ Total Runtime: {runtime_hours:.2f} hours")
        logger.info(f"📁 Files Saved: {self.stats['total_files_saved']}")
        logger.info(f"📊 Total Rows: {self.stats['total_rows_collected']:,}")
        logger.info(f"✅ Successful Uploads: {self.stats['successful_uploads']}")
        logger.info(f"❌ Failed Uploads: {self.stats['failed_uploads']}")
        
        if self.stats['total_files_saved'] > 0:
            avg_rows = self.stats['total_rows_collected'] / self.stats['total_files_saved']
            logger.info(f"📈 Average Rows/File: {avg_rows:.1f}")
        
        if runtime_hours > 0:
            files_per_hour = self.stats['total_files_saved'] / runtime_hours
            rows_per_hour = self.stats['total_rows_collected'] / runtime_hours
            logger.info(f"📊 Collection Rate: {files_per_hour:.1f} files/hour, {rows_per_hour:,.0f} rows/hour")
        
        storage_status = self.storage.get_storage_status()
        logger.info(f"💾 Final Storage: Memory={storage_status['memory_files']}, "
                   f"Local={storage_status['local_files']}, Drive={storage_status['drive_files']}")
        
        if self.stats['errors']:
            logger.warning(f"⚠️ Total Errors: {len(self.stats['errors'])}")
            for i, error in enumerate(self.stats['errors'][-5:], 1):  # Show last 5 errors
                logger.warning(f"   {i}. {error}")
        else:
            logger.info("✅ No errors encountered!")
        
        logger.info("=" * 60)
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get comprehensive collection summary"""
        summary = self.stats.copy()
        summary['storage_status'] = self.storage.get_storage_status()
        
        if self.stats['start_time'] and self.stats['end_time']:
            runtime = self.stats['end_time'] - self.stats['start_time']
            summary['runtime_hours'] = runtime.total_seconds() / 3600
            summary['runtime_str'] = str(runtime)
        
        return summary
    
    def stop_collection(self):
        """Stop the collection gracefully"""
        logger.info("🛑 Stop requested...")
        self.stop_flag.set()

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Data Scraper')
    parser.add_argument('--hours', type=float, help='Run for specified hours (continuous if not set)')
    parser.add_argument('--cycle-interval', type=int, default=5, 
                       help='Minutes between collection cycles (default: 5)')
    parser.add_argument('--symbols', nargs='+', 
                       default=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'],
                       help='Symbols to collect')
    parser.add_argument('--intervals', nargs='+', 
                       default=['1m', '5m', '15m', '1h'],
                       help='Intervals to collect')
    parser.add_argument('--memory-only', action='store_true',
                       help='Use memory-only storage (for testing)')
    parser.add_argument('--local-backup', type=str,
                       help='Local backup directory')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('production_scraper.log')
        ]
    )
    
    # Load config
    from src.config import GOOGLE_DRIVE_FOLDER_ID, BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET
    
    # Create scraper
    scraper = ProductionDataScraper(
        symbols=args.symbols,
        intervals=args.intervals,
        drive_folder_id=GOOGLE_DRIVE_FOLDER_ID,
        local_backup_dir=args.local_backup,
        memory_only=args.memory_only
    )
    
    # Run collection
    if args.hours:
        results = scraper.run_timed_collection(args.hours, args.cycle_interval)
    else:
        results = scraper.run_continuous_collection(args.cycle_interval)
    
    # Print final summary
    print("\n" + "="*60)
    print("COLLECTION COMPLETED")
    print("="*60)
    print(f"Files saved: {results['total_files_saved']}")
    print(f"Rows collected: {results['total_rows_collected']:,}")
    print(f"Success rate: {results['successful_uploads']}/{results['successful_uploads'] + results['failed_uploads']}")
    
    if results.get('runtime_hours'):
        print(f"Runtime: {results['runtime_hours']:.2f} hours")
    
    storage_status = results['storage_status']
    print(f"Storage: Memory={storage_status['memory_files']}, "
          f"Local={storage_status['local_files']}, Drive={storage_status['drive_files']}")

if __name__ == "__main__":
    main()

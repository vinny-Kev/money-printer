"""
Production-Ready Money Printer Trading Bot
Main entry point with complete integration of data collection, training, and trading
"""
import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import production components
from src.data_collector.production_data_scraper import ProductionDataScraper
from src.model_training.production_trainer import ProductionModelTrainer
from src.storage.enhanced_storage_manager import EnhancedStorageManager
from src.binance_wrapper import EnhancedBinanceClient
from src.config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('money_printer.log')
    ]
)
logger = logging.getLogger(__name__)

def show_status():
    """Show comprehensive system status including real Binance balance"""
    logger.info("📊 Money Printer System Status")
    logger.info("=" * 50)
    
    # Binance Connection and Balance
    try:
        logger.info("🔗 Testing Binance connection...")
        binance = EnhancedBinanceClient(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_SECRET_KEY,
            testnet=BINANCE_TESTNET
        )
        
        if binance.test_connection():
            logger.info("✅ Binance: Connected")
            
            # Get real account info
            account_info = binance.get_account_info()
            if account_info:
                balances = account_info['balances']
                non_zero_balances = [b for b in balances if b['total'] > 0]
                
                logger.info(f"💰 Account Type: {account_info['account_type']}")
                logger.info(f"🔄 Can Trade: {account_info['can_trade']}")
                logger.info(f"💵 Assets with Balance: {len(non_zero_balances)}")
                
                logger.info("💰 Current Balances:")
                for balance in sorted(non_zero_balances, key=lambda x: x['total'], reverse=True):
                    logger.info(f"   {balance['asset']}: {balance['total']:.8f} "
                               f"(Free: {balance['free']:.8f}, Locked: {balance['locked']:.8f})")
                
                # Calculate total value in USDT (simplified)
                usdt_balance = next((b['total'] for b in balances if b['asset'] == 'USDT'), 0)
                logger.info(f"💵 USDT Available for Trading: ${usdt_balance:.2f}")
            else:
                logger.warning("⚠️ Could not retrieve account information")
        else:
            logger.error("❌ Binance: Connection failed")
    except Exception as e:
        logger.error(f"❌ Binance error: {e}")
    
    # Google Drive Status
    try:
        logger.info("\n☁️ Testing Google Drive connection...")
        service_key_path = "secrets/service_account.json"
        if os.path.exists(service_key_path):
            from src.drive_manager import EnhancedDriveManager
            drive_manager = EnhancedDriveManager()
            files = drive_manager.list_files_in_folder()
            logger.info(f"✅ Google Drive: Connected ({len(files)} files)")
        else:
            logger.warning("⚠️ Google Drive: Service account key missing")
            logger.info("💡 Add secrets/service_account.json for Drive integration")
    except Exception as e:
        logger.error(f"❌ Google Drive error: {e}")
    
    # Storage Status
    try:
        logger.info("\n📁 Checking local storage...")
        storage_manager = EnhancedStorageManager(
            drive_folder_id=GOOGLE_DRIVE_FOLDER_ID,
            local_backup_dir="data/production",
            memory_only=False
        )
        
        storage_status = storage_manager.get_storage_status()
        logger.info(f"💾 Local files: {storage_status['local_files']}")
        logger.info(f"☁️ Drive files: {storage_status['drive_files']}")
        logger.info(f"🧠 Memory files: {storage_status['memory_files']}")
        
        if storage_status['total_memory_rows'] > 0:
            logger.info(f"📊 Total data rows in memory: {storage_status['total_memory_rows']:,}")
    except Exception as e:
        logger.error(f"❌ Storage error: {e}")
    
    # Model Status
    try:
        logger.info("\n🤖 Checking trained models...")
        model_dir = Path("models")
        if model_dir.exists():
            model_files = list(model_dir.glob("production_model_*.joblib"))
            logger.info(f"🎯 Available models: {len(model_files)}")
            
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                model_age = datetime.now() - datetime.fromtimestamp(latest_model.stat().st_mtime)
                logger.info(f"📅 Latest model: {latest_model.name}")
                logger.info(f"⏰ Model age: {model_age}")
            else:
                logger.warning("⚠️ No trained models found")
        else:
            logger.warning("⚠️ Models directory not found")
    except Exception as e:
        logger.error(f"❌ Model check error: {e}")
    
    logger.info("\n" + "=" * 50)

def run_data_collection(hours: float = None, symbols: list = None, intervals: list = None):
    """Run production data collection"""
    logger.info("🚀 Starting Production Data Collection")
    
    # Set defaults
    symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    intervals = intervals or ['1m', '5m', '15m', '1h']
    
    # Create storage manager (with Drive fallback)
    storage_manager = EnhancedStorageManager(
        drive_folder_id=GOOGLE_DRIVE_FOLDER_ID,
        local_backup_dir="data/production",
        memory_only=False
    )
    
    # Create scraper
    scraper = ProductionDataScraper(
        symbols=symbols,
        intervals=intervals,
        drive_folder_id=GOOGLE_DRIVE_FOLDER_ID,
        local_backup_dir="data/production",
        memory_only=False
    )
    
    if hours:
        logger.info(f"📅 Collecting data for {hours} hours...")
        results = scraper.run_timed_collection(hours, cycle_interval_minutes=5)
    else:
        logger.info("♾️ Collecting data continuously...")
        results = scraper.run_continuous_collection(cycle_interval_minutes=5)
    
    # Report results
    logger.info("📊 Collection Summary:")
    logger.info(f"   📁 Files saved: {results['total_files_saved']}")
    logger.info(f"   📊 Rows collected: {results['total_rows_collected']:,}")
    logger.info(f"   ✅ Success rate: {results['successful_uploads']}/{results['successful_uploads'] + results['failed_uploads']}")
    
    return results

def run_model_training():
    """Run production model training"""
    logger.info("🤖 Starting Production Model Training")
    
    # Create storage manager
    storage_manager = EnhancedStorageManager(
        drive_folder_id=GOOGLE_DRIVE_FOLDER_ID,
        local_backup_dir="data/production",
        memory_only=False
    )
    
    # Create trainer
    trainer = ProductionModelTrainer(
        storage_manager=storage_manager,
        min_rows_total=500,
        min_rows_per_symbol=50,
        min_time_span_hours=6,
        model_output_dir="models"
    )
    
    # Run training
    results = trainer.run_full_training_pipeline()
    
    if results['pipeline_success']:
        training = results['training_results']
        logger.info("✅ Training completed successfully!")
        logger.info(f"   📊 Training samples: {training['training_samples']:,}")
        logger.info(f"   🧪 Test samples: {training['test_samples']:,}")
        logger.info(f"   ⚙️ Features: {training['features_used']}")
        logger.info(f"   📈 Test R²: {training['performance']['test_r2']:.4f}")
        logger.info(f"   💾 Model saved: {training['model_path']}")
    else:
        logger.error("❌ Training failed!")
        if results.get('data_validation'):
            validation = results['data_validation']
            if not validation.get('is_sufficient', True):
                logger.error("💡 Insufficient data for training:")
                for issue in validation.get('issues', []):
                    logger.error(f"   • {issue}")
    
    return results

def run_system_test():
    """Run comprehensive system test"""
    logger.info("🧪 Running Comprehensive System Test")
    
    from production_system_test import ProductionSystemTester
    tester = ProductionSystemTester()
    results = tester.run_comprehensive_test()
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Money Printer Production Trading Bot')
    parser.add_argument('command', choices=['status', 'collect', 'train', 'test'],
                       help='Command to execute')
    parser.add_argument('--hours', type=float,
                       help='Hours to run data collection (continuous if not specified)')
    parser.add_argument('--symbols', nargs='+',
                       default=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'],
                       help='Symbols to collect/trade')
    parser.add_argument('--intervals', nargs='+',
                       default=['1m', '5m', '15m', '1h'],
                       help='Intervals to collect')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Money Printer v2.0 - Production Ready")
    logger.info(f"📅 Started at: {datetime.now()}")
    
    try:
        if args.command == 'status':
            show_status()
        
        elif args.command == 'collect':
            run_data_collection(args.hours, args.symbols, args.intervals)
        
        elif args.command == 'train':
            run_model_training()
        
        elif args.command == 'test':
            run_system_test()
        
        else:
            logger.error(f"❌ Unknown command: {args.command}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

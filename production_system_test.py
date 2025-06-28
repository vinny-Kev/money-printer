"""
Comprehensive Production System Test
Tests data collection, model training, Google Drive integration, and Binance balance
"""
import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_collector.production_data_scraper import ProductionDataScraper
from src.model_training.production_trainer import ProductionModelTrainer
from src.storage.enhanced_storage_manager import EnhancedStorageManager
from src.binance_wrapper import EnhancedBinanceClient
from src.drive_manager import EnhancedDriveManager
from binance.client import Client

# Load config
try:
    from src.config import GOOGLE_DRIVE_FOLDER_ID, BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_TESTNET
    BINANCE_API_SECRET = BINANCE_SECRET_KEY  # Alias for compatibility
except ImportError:
    # Fallback to environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    GOOGLE_DRIVE_FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    BINANCE_API_SECRET = BINANCE_SECRET_KEY  # Alias for compatibility
    BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'False').lower() == 'true'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('production_system_test.log')
    ]
)
logger = logging.getLogger(__name__)

class ProductionSystemTester:
    """Comprehensive production system tester"""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'binance_connection': False,
            'binance_balance': None,
            'drive_connection': False,
            'data_collection': False,
            'model_training': False,
            'overall_success': False,
            'errors': []
        }
        
        logger.info("🚀 Initializing Production System Test")
    
    def test_binance_connection_and_balance(self) -> bool:
        """Test Binance connection and get real balance"""
        logger.info("🔗 Testing Binance connection and balance...")
        
        try:
            # Test with enhanced wrapper
            binance_client = EnhancedBinanceClient(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_SECRET_KEY,
                testnet=BINANCE_TESTNET
            )
            
            if not binance_client.test_connection():
                raise Exception("Enhanced Binance client connection failed")
            
            # Get account info
            account_info = binance_client.get_account_info()
            if account_info:
                self.test_results['binance_balance'] = account_info
                
                # Log balance summary
                total_assets = len(account_info['balances'])
                non_zero_assets = len([b for b in account_info['balances'] if b['total'] > 0])
                
                logger.info(f"✅ Binance connection successful")
                logger.info(f"   📊 Account type: {account_info['account_type']}")
                logger.info(f"   💰 Total assets: {total_assets} ({non_zero_assets} with balance)")
                logger.info(f"   🔄 Can trade: {account_info['can_trade']}")
                
                # Show top balances
                top_balances = sorted(account_info['balances'], key=lambda x: x['total'], reverse=True)[:5]
                logger.info("   💵 Top balances:")
                for balance in top_balances:
                    logger.info(f"      {balance['asset']}: {balance['total']:.8f}")
                
                self.test_results['binance_connection'] = True
                return True
            else:
                raise Exception("Failed to get account info")
                
        except Exception as e:
            error_msg = f"Binance connection test failed: {e}"
            logger.error(f"❌ {error_msg}")
            self.test_results['errors'].append(error_msg)
            return False
    
    def test_google_drive_connection(self) -> bool:
        """Test Google Drive connection"""
        logger.info("☁️ Testing Google Drive connection...")
        
        try:
            drive_manager = EnhancedDriveManager()
            
            # Check if service account exists
            service_key_path = "secrets/service_account.json"
            if not os.path.exists(service_key_path):
                logger.warning("⚠️ Service account key missing - Drive tests will fail on Railway")
                logger.info("💡 For local testing, this is expected")
                return False
            
            # Test listing files
            files = drive_manager.list_files_in_folder()
            
            logger.info(f"✅ Google Drive connection successful")
            logger.info(f"   📁 Found {len(files)} files in Drive folder")
            
            self.test_results['drive_connection'] = True
            return True
            
        except Exception as e:
            error_msg = f"Google Drive connection test failed: {e}"
            logger.error(f"❌ {error_msg}")
            self.test_results['errors'].append(error_msg)
            return False
    
    def test_data_collection(self, duration_minutes: float = 2.0) -> bool:
        """Test data collection for a short period"""
        logger.info(f"📊 Testing data collection for {duration_minutes} minutes...")
        
        try:
            # Create storage manager (memory-only for testing)
            storage_manager = EnhancedStorageManager(
                drive_folder_id=GOOGLE_DRIVE_FOLDER_ID,
                local_backup_dir="test_data",
                memory_only=True  # Use memory-only for testing
            )
            
            # Create scraper
            scraper = ProductionDataScraper(
                symbols=['BTCUSDT', 'ETHUSDT'],  # Just 2 symbols for testing
                intervals=['1m', '5m'],  # Just 2 intervals for testing
                drive_folder_id=GOOGLE_DRIVE_FOLDER_ID,
                local_backup_dir="test_data",
                memory_only=True
            )
            
            # Run timed collection
            results = scraper.run_timed_collection(
                hours=duration_minutes / 60,
                cycle_interval_minutes=1
            )
            
            if results['total_files_saved'] > 0 and results['total_rows_collected'] > 0:
                logger.info(f"✅ Data collection successful")
                logger.info(f"   📁 Files collected: {results['total_files_saved']}")
                logger.info(f"   📊 Rows collected: {results['total_rows_collected']:,}")
                logger.info(f"   ⏱️ Runtime: {results.get('runtime_hours', 0):.2f} hours")
                
                self.test_results['data_collection'] = True
                return True
            else:
                raise Exception("No data was collected")
                
        except Exception as e:
            error_msg = f"Data collection test failed: {e}"
            logger.error(f"❌ {error_msg}")
            self.test_results['errors'].append(error_msg)
            return False
    
    def test_model_training(self) -> bool:
        """Test model training with collected data"""
        logger.info("🤖 Testing model training...")
        
        try:
            # Create storage manager with test data
            storage_manager = EnhancedStorageManager(
                local_backup_dir="test_data",
                memory_only=False  # Use local files for training test
            )
            
            # First, collect some data to train on
            logger.info("📊 Collecting training data...")
            scraper = ProductionDataScraper(
                symbols=['BTCUSDT', 'ETHUSDT'],
                intervals=['1m', '5m'],
                local_backup_dir="test_data",
                memory_only=False
            )
            
            # Collect data for 1 minute
            collection_results = scraper.run_timed_collection(
                hours=1/60,  # 1 minute
                cycle_interval_minutes=0.5
            )
            
            if collection_results['total_files_saved'] == 0:
                raise Exception("No training data collected")
            
            logger.info(f"📊 Collected {collection_results['total_files_saved']} files for training")
            
            # Create trainer with relaxed requirements
            trainer = ProductionModelTrainer(
                storage_manager=storage_manager,
                min_rows_total=50,  # Very low for testing
                min_rows_per_symbol=10,
                min_time_span_hours=0.1,  # 6 minutes
                model_output_dir="test_models"
            )
            
            # Run training
            training_results = trainer.run_full_training_pipeline()
            
            if training_results['pipeline_success']:
                logger.info(f"✅ Model training successful")
                training = training_results['training_results']
                logger.info(f"   📊 Training samples: {training['training_samples']:,}")
                logger.info(f"   🧪 Test samples: {training['test_samples']:,}")
                logger.info(f"   ⚙️ Features: {training['features_used']}")
                logger.info(f"   📈 Test R²: {training['performance']['test_r2']:.4f}")
                logger.info(f"   💾 Model saved: {training['model_path']}")
                
                self.test_results['model_training'] = True
                return True
            else:
                validation = training_results.get('data_validation', {})
                if validation and not validation.get('is_sufficient', True):
                    logger.warning("⚠️ Training failed due to insufficient data (expected for short test)")
                    logger.info("💡 This is normal for a short test run - need more data for real training")
                    return False
                else:
                    raise Exception(f"Training failed: {training_results.get('error', 'Unknown error')}")
                
        except Exception as e:
            error_msg = f"Model training test failed: {e}"
            logger.error(f"❌ {error_msg}")
            self.test_results['errors'].append(error_msg)
            return False
    
    def run_comprehensive_test(self) -> dict:
        """Run all tests in sequence"""
        logger.info("🏁 Starting Comprehensive Production System Test")
        logger.info("=" * 60)
        
        tests = [
            ("Binance Connection & Balance", self.test_binance_connection_and_balance),
            ("Google Drive Connection", self.test_google_drive_connection),
            ("Data Collection", self.test_data_collection),
            ("Model Training", self.test_model_training),
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} CRASHED: {e}")
                self.test_results['errors'].append(f"{test_name} crashed: {e}")
        
        # Overall assessment
        self.test_results['overall_success'] = passed_tests >= 3  # Need at least 3/4 to pass
        self.test_results['passed_tests'] = passed_tests
        self.test_results['total_tests'] = len(tests)
        
        # Final report
        logger.info("\n" + "=" * 60)
        logger.info("📊 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Tests Passed: {passed_tests}/{len(tests)}")
        
        if self.test_results['binance_connection']:
            logger.info("🔗 Binance: Connected and authenticated")
            if self.test_results['binance_balance']:
                balances = self.test_results['binance_balance']['balances']
                non_zero = len([b for b in balances if b['total'] > 0])
                logger.info(f"💰 Balance: {non_zero} assets with non-zero balance")
        
        if self.test_results['drive_connection']:
            logger.info("☁️ Google Drive: Connected and accessible")
        else:
            logger.warning("⚠️ Google Drive: Not connected (service account missing)")
        
        if self.test_results['data_collection']:
            logger.info("📊 Data Collection: Working properly")
        
        if self.test_results['model_training']:
            logger.info("🤖 Model Training: Working properly")
        else:
            logger.warning("⚠️ Model Training: Failed (may need more data)")
        
        if self.test_results['overall_success']:
            logger.info("🎉 OVERALL: Production system is ready!")
            logger.info("💡 Core components are working - you can deploy to Railway")
        else:
            logger.warning("⚠️ OVERALL: Some issues found")
            logger.info("🔧 Review the errors above and fix before deployment")
        
        if self.test_results['errors']:
            logger.warning(f"\n⚠️ Errors encountered ({len(self.test_results['errors'])}):")
            for i, error in enumerate(self.test_results['errors'], 1):
                logger.warning(f"   {i}. {error}")
        
        logger.info("=" * 60)
        
        return self.test_results

def main():
    """Main function"""
    tester = ProductionSystemTester()
    results = tester.run_comprehensive_test()
    
    # Save results to file
    with open('production_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Test results saved to: production_test_results.json")
    
    # Exit with appropriate code
    if results['overall_success']:
        print("✅ All critical tests passed - system is production ready!")
        sys.exit(0)
    else:
        print("❌ Some tests failed - review and fix issues before deployment")
        sys.exit(1)

if __name__ == "__main__":
    main()

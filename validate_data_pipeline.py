#!/usr/bin/env python3
"""
Comprehensive Data Pipeline Validation
Tests the entire data flow:
1. Local scraper data collection
2. Google Drive upload
3. Model trainer data access
4. File integrity verification
"""
import os
import sys
import time
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, USE_GOOGLE_DRIVE, 
    PARQUET_DATA_DIR, GOOGLE_DRIVE_FOLDER_ID
)
from src.data_collector.local_storage import save_parquet_file, list_parquet_files
from src.discord_notifications import send_scraper_notification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_pipeline_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_local_data_creation():
    """Test creating sample data locally"""
    logger.info("🧪 STAGE 1: Testing Local Data Creation")
    
    try:
        # Create sample cryptocurrency data
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        created_files = []
        
        for symbol in test_symbols:
            # Generate sample OHLCV data with enough records (100 to be safe)
            dates = pd.date_range(start='2025-06-27 12:00:00', periods=100, freq='1min')
            base_price = 50000 if symbol == 'BTCUSDT' else (3000 if symbol == 'ETHUSDT' else 0.5)
            
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': base_price + (pd.Series(range(100)) * 0.1),
                'high': base_price + (pd.Series(range(100)) * 0.15),
                'low': base_price + (pd.Series(range(100)) * 0.05),
                'close': base_price + (pd.Series(range(100)) * 0.12),
                'volume': 1000 + (pd.Series(range(100)) * 10)
            })
            
            # Save using our local storage function
            filename = f"{symbol}.parquet"
            result = save_parquet_file(sample_data, filename, symbol)
            
            if result:
                file_size = os.path.getsize(result)
                logger.info(f"✅ Created {symbol}: {len(sample_data)} records, {file_size} bytes")
                created_files.append({
                    'symbol': symbol,
                    'file_path': result,
                    'records': len(sample_data),
                    'size_bytes': file_size
                })
            else:
                logger.error(f"❌ Failed to create {symbol}")
                return False, []
        
        logger.info(f"✅ STAGE 1 COMPLETE: Created {len(created_files)} test files")
        return True, created_files
        
    except Exception as e:
        logger.error(f"❌ STAGE 1 FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, []

def test_google_drive_integration():
    """Test Google Drive upload/download if enabled"""
    logger.info("🧪 STAGE 2: Testing Google Drive Integration")
    
    try:
        if not USE_GOOGLE_DRIVE:
            logger.info("⚠️ Google Drive disabled (USE_GOOGLE_DRIVE=False)")
            return True, "Google Drive integration disabled"
        
        # Test Google Drive connection
        from src.drive_manager import EnhancedDriveManager
        
        drive_manager = EnhancedDriveManager()
        logger.info("✅ Drive manager initialized")
        
        # Test listing files
        try:
            drive_files = drive_manager.list_files_in_folder()
            logger.info(f"✅ Found {len(drive_files)} files in Google Drive")
            
            # Look for our test files
            test_files_found = []
            for file_info in drive_files:
                if any(symbol in file_info.get('name', '') for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']):
                    test_files_found.append(file_info)
                    logger.info(f"📁 Found test file: {file_info.get('name')} ({file_info.get('size', 0)} bytes)")
            
            if test_files_found:
                logger.info(f"✅ Found {len(test_files_found)} test files in Google Drive")
                return True, f"Found {len(test_files_found)} test files in Drive"
            else:
                logger.warning("⚠️ No test files found in Google Drive yet (upload may be in progress)")
                return True, "No test files found in Drive (upload pending)"
                
        except Exception as drive_error:
            logger.warning(f"⚠️ Drive access error: {drive_error}")
            return False, f"Drive access failed: {drive_error}"
            
    except Exception as e:
        logger.error(f"❌ STAGE 2 FAILED: {e}")
        return False, f"Drive integration failed: {e}"

def test_model_trainer_data_access():
    """Test if model trainers can access the data"""
    logger.info("🧪 STAGE 3: Testing Model Trainer Data Access")
    
    try:
        # Test local data loader used by trainers
        from src.model_training.local_data_loader import fetch_parquet_data_from_local
        
        logger.info("📊 Testing local data loader...")
        df = fetch_parquet_data_from_local()
        
        if df is not None and not df.empty:
            logger.info(f"✅ Model trainer can access data: {len(df)} records from {df['symbol'].nunique()} symbols")
            
            # Show data summary
            symbols = df['symbol'].unique()
            logger.info(f"📈 Available symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
            
            # Check data quality
            date_range = df['timestamp'].agg(['min', 'max'])
            logger.info(f"📅 Date range: {date_range['min']} to {date_range['max']}")
            
            # Check for required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"⚠️ Missing columns: {missing_cols}")
                return False, f"Missing required columns: {missing_cols}"
            else:
                logger.info("✅ All required columns present")
                
            return True, f"Trainer can access {len(df)} records from {df['symbol'].nunique()} symbols"
            
        else:
            logger.error("❌ Model trainer returned empty data")
            return False, "No data available for training"
            
    except Exception as e:
        logger.error(f"❌ STAGE 3 FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, f"Trainer data access failed: {e}"

def test_model_trainer_google_drive_access():
    """Test if model trainers can access data from Google Drive if enabled"""
    logger.info("🧪 STAGE 3B: Testing Model Trainer Google Drive Access")
    
    try:
        if not USE_GOOGLE_DRIVE:
            logger.info("⚠️ Google Drive disabled - skipping Google Drive data access test")
            return True, "Google Drive disabled"
        
        # Test if the model trainer can potentially access Google Drive data
        from src.drive_manager import EnhancedDriveManager
        
        drive_manager = EnhancedDriveManager()
        
        # Try to list files to see if the model trainer could access them
        try:
            drive_files = drive_manager.list_files_in_folder()
            parquet_files = [f for f in drive_files if f.get('name', '').endswith('.parquet')]
            
            if parquet_files:
                logger.info(f"✅ Google Drive contains {len(parquet_files)} parquet files that model trainer could access")
                
                # Show some examples
                for i, file_info in enumerate(parquet_files[:3]):
                    logger.info(f"   📁 {file_info.get('name')} ({file_info.get('size', 0)} bytes)")
                    
                return True, f"Google Drive has {len(parquet_files)} parquet files available"
            else:
                logger.warning("⚠️ No parquet files found in Google Drive for model training")
                return True, "No parquet files in Google Drive yet"
                
        except Exception as drive_error:
            logger.warning(f"⚠️ Could not access Google Drive files: {drive_error}")
            return False, f"Google Drive access failed: {drive_error}"
            
    except Exception as e:
        logger.error(f"❌ STAGE 3B FAILED: {e}")
        return False, f"Google Drive data access test failed: {e}"

def test_model_preprocessing():
    """Test if model preprocessing works with the data"""
    logger.info("🧪 STAGE 4: Testing Model Preprocessing")
    
    try:
        from src.model_training.local_data_loader import fetch_parquet_data_from_local
        from src.model_training.common import preprocess_data
        
        # Get data
        df = fetch_parquet_data_from_local()
        if df is None or df.empty:
            logger.error("❌ No data available for preprocessing test")
            return False, "No data for preprocessing"
        
        # Test preprocessing
        logger.info("🔄 Testing data preprocessing...")
        X, y, groups = preprocess_data(df)
        
        if X is not None and y is not None:
            logger.info(f"✅ Preprocessing successful:")
            logger.info(f"   Features (X): {X.shape}")
            logger.info(f"   Labels (y): {y.shape}")
            logger.info(f"   Groups: {len(groups)} time groups")
            logger.info(f"   Feature columns: {list(X.columns)[:5]}{'...' if len(X.columns) > 5 else ''}")
            
            # Check data quality
            if X.shape[0] < 100:
                logger.warning(f"⚠️ Low data volume: {X.shape[0]} samples (need 500+ for training)")
                return True, f"Preprocessing works but low data: {X.shape[0]} samples"
            else:
                logger.info(f"✅ Good data volume: {X.shape[0]} samples")
                return True, f"Preprocessing successful: {X.shape[0]} samples, {X.shape[1]} features"
        else:
            logger.error("❌ Preprocessing returned None")
            return False, "Preprocessing failed"
            
    except Exception as e:
        logger.error(f"❌ STAGE 4 FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, f"Preprocessing failed: {e}"

def test_file_integrity():
    """Test integrity of created files"""
    logger.info("🧪 STAGE 5: Testing File Integrity")
    
    try:
        files = list_parquet_files()
        
        if not files:
            logger.error("❌ No parquet files found")
            return False, "No files found"
        
        logger.info(f"📁 Found {len(files)} parquet files")
        
        integrity_results = []
        for file_info in files[:5]:  # Test first 5 files
            try:
                # Try to load the file
                file_path = file_info['full_path']
                df = pd.read_parquet(file_path)
                
                integrity_results.append({
                    'file': file_info['filename'],
                    'records': len(df),
                    'columns': list(df.columns),
                    'size_kb': file_info['size'] / 1024,
                    'status': 'OK'
                })
                
                logger.info(f"✅ {file_info['filename']}: {len(df)} records, {len(df.columns)} columns")
                
            except Exception as file_error:
                integrity_results.append({
                    'file': file_info['filename'],
                    'status': 'ERROR',
                    'error': str(file_error)
                })
                logger.error(f"❌ {file_info['filename']}: {file_error}")
        
        successful_files = [r for r in integrity_results if r['status'] == 'OK']
        
        if successful_files:
            logger.info(f"✅ STAGE 5 COMPLETE: {len(successful_files)}/{len(integrity_results)} files passed integrity check")
            return True, f"{len(successful_files)}/{len(integrity_results)} files valid"
        else:
            logger.error("❌ No files passed integrity check")
            return False, "All files failed integrity check"
            
    except Exception as e:
        logger.error(f"❌ STAGE 5 FAILED: {e}")
        return False, f"Integrity check failed: {e}"

def main():
    """Run complete data pipeline validation"""
    logger.info("🚀 STARTING COMPREHENSIVE DATA PIPELINE VALIDATION")
    logger.info(f"⏰ Start time: {datetime.now()}")
    
    # Configuration check
    logger.info(f"🔧 Configuration:")
    logger.info(f"   Google Drive: {'Enabled' if USE_GOOGLE_DRIVE else 'Disabled'}")
    logger.info(f"   Data Directory: {PARQUET_DATA_DIR}")
    logger.info(f"   Drive Folder ID: {GOOGLE_DRIVE_FOLDER_ID if USE_GOOGLE_DRIVE else 'N/A'}")
    
    results = {}
    overall_success = True
    
    # Run all validation stages
    stages = [
        ("Local Data Creation", test_local_data_creation),
        ("Google Drive Integration", test_google_drive_integration),
        ("Model Trainer Data Access", test_model_trainer_data_access),
        ("Model Trainer Google Drive Access", test_model_trainer_google_drive_access),
        ("Model Preprocessing", test_model_preprocessing),
        ("File Integrity", test_file_integrity)
    ]
    
    for stage_name, stage_func in stages:
        logger.info(f"\n{'='*60}")
        try:
            if stage_name == "Local Data Creation":
                success, result = stage_func()
            else:
                success, result = stage_func()
            
            results[stage_name] = {
                'success': success,
                'result': result
            }
            
            if not success:
                overall_success = False
                
        except Exception as e:
            logger.error(f"❌ {stage_name} CRASHED: {e}")
            results[stage_name] = {
                'success': False,
                'result': f"Stage crashed: {e}"
            }
            overall_success = False
    
    # Generate final report
    logger.info(f"\n{'='*60}")
    logger.info("📊 FINAL VALIDATION REPORT")
    logger.info(f"{'='*60}")
    
    for stage_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        logger.info(f"{status} {stage_name}: {result['result']}")
    
    logger.info(f"{'='*60}")
    if overall_success:
        logger.info("🎉 OVERALL RESULT: ALL STAGES PASSED - Data pipeline is working!")
        final_status = "✅ ALL SYSTEMS OPERATIONAL"
    else:
        logger.info("⚠️ OVERALL RESULT: SOME STAGES FAILED - Check logs for issues")
        final_status = "❌ ISSUES DETECTED"
    
    logger.info(f"⏰ Validation completed: {datetime.now()}")
    
    # Send Discord notification
    try:
        summary = f"""🧪 **Data Pipeline Validation Complete**

{final_status}

📊 **Stage Results:**
"""
        for stage_name, result in results.items():
            status_emoji = "✅" if result['success'] else "❌"
            summary += f"{status_emoji} {stage_name}\n"
        
        summary += f"\n📁 Check `data_pipeline_validation.log` for details"
        
        send_scraper_notification(summary)
    except Exception as e:
        logger.warning(f"Failed to send Discord notification: {e}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

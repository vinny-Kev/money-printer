"""
Test Google Drive fallback functionality in the data loader
"""
import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_drive_fallback():
    """Test that the data loader can fall back to Google Drive when local files are missing"""
    
    logger.info("🧪 Testing Google Drive fallback functionality...")
    
    # Create a temporary directory to simulate empty local storage
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"📁 Using temporary directory: {temp_path}")
        
        # Import the data loader with our custom temp directory
        from src.model_training.local_data_loader import LocalDataLoader, fetch_parquet_data_from_local
        
        # Test 1: LocalDataLoader with empty directory
        logger.info("\n📋 Test 1: LocalDataLoader with empty directory")
        loader = LocalDataLoader(data_directory=str(temp_path))
        
        symbols = loader.list_available_symbols()
        logger.info(f"   Symbols found in empty directory: {len(symbols)}")
        
        # Test 2: Try to load all data (should be empty)
        logger.info("\n📋 Test 2: Loading data from empty directory")
        df_empty = loader.load_all_data()
        logger.info(f"   Rows loaded from empty directory: {len(df_empty)}")
        
        # Test 3: Test the enhanced fetch function (with Drive fallback)
        logger.info("\n📋 Test 3: Testing fetch_parquet_data_from_local with Drive fallback")
        
        # Temporarily override the data directory in config to point to our empty temp dir
        try:
            import src.config
            original_dir = getattr(src.config, 'PARQUET_DATA_DIR', None)
            src.config.PARQUET_DATA_DIR = str(temp_path)
            
            # Test the fetch function
            try:
                df_with_fallback = fetch_parquet_data_from_local()
                logger.info(f"   ✅ Successfully loaded {len(df_with_fallback)} rows with fallback")
                
                if len(df_with_fallback) > 0:
                    logger.info(f"   📊 Data columns: {list(df_with_fallback.columns)}")
                    if 'symbol' in df_with_fallback.columns:
                        unique_symbols = df_with_fallback['symbol'].nunique()
                        logger.info(f"   🪙 Unique symbols: {unique_symbols}")
                else:
                    logger.warning("   ⚠️ No data loaded even with fallback")
                    
            except Exception as e:
                logger.error(f"   ❌ Fallback failed: {e}")
                
        finally:
            # Restore original config
            if original_dir:
                src.config.PARQUET_DATA_DIR = original_dir
    
    # Test 4: Check if we have actual local data
    logger.info("\n📋 Test 4: Checking actual local data availability")
    from src.model_training.local_data_loader import LocalDataLoader
    real_loader = LocalDataLoader()  # Use default directory
    real_symbols = real_loader.list_available_symbols()
    logger.info(f"   Real local symbols available: {len(real_symbols)}")
    if real_symbols:
        logger.info(f"   Sample symbols: {real_symbols[:5]}")
        
        # Test loading one symbol
        sample_symbol = real_symbols[0] if real_symbols else None
        if sample_symbol:
            sample_df = real_loader.load_symbol_data(sample_symbol)
            if sample_df is not None:
                logger.info(f"   Sample data for {sample_symbol}: {len(sample_df)} rows")
    
    # Test 5: Check Google Drive files directly
    logger.info("\n📋 Test 5: Checking Google Drive files directly")
    try:
        from src.drive_manager import EnhancedDriveManager
        drive_manager = EnhancedDriveManager()
        if drive_manager.authenticated:
            files = drive_manager.list_files_in_folder()
            parquet_files = [f for f in files if f.get('name', '').endswith('.parquet')]
            logger.info(f"   ☁️ Parquet files in Google Drive: {len(parquet_files)}")
            
            if parquet_files:
                logger.info("   📁 Recent parquet files:")
                for file in parquet_files[:5]:  # Show first 5
                    name = file.get('name', 'Unknown')
                    size = file.get('size', 'Unknown')
                    modified = file.get('modifiedTime', 'Unknown')
                    logger.info(f"      {name} ({size} bytes, {modified})")
        else:
            logger.warning("   ❌ Google Drive not authenticated")
    except Exception as e:
        logger.error(f"   ❌ Error checking Google Drive: {e}")

if __name__ == "__main__":
    test_drive_fallback()

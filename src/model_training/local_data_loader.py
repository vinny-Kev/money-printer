import os
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class LocalDataLoader:
    """
    Local data loader that replaces the bucket-based data fetching.
    Loads parquet files from local storage for model training.
    """
    
    def __init__(self, data_directory: str = None):
        """
        Initialize the local data loader.
        
        Args:
            data_directory: Path to the directory containing parquet files.
                           If None, uses configuration from config.py.
        """
        if data_directory is None:
            # Use configuration from config.py
            try:
                from src.config import PARQUET_DATA_DIR
                self.data_directory = Path(PARQUET_DATA_DIR)
            except ImportError:
                # Fallback to default path
                self.data_directory = Path(__file__).parent.parent.parent / "data" / "scraped_data" / "parquet_files"
        else:
            self.data_directory = Path(data_directory)
        
        self.data_directory = self.data_directory.resolve()
        logger.info(f"LocalDataLoader initialized with data directory: {self.data_directory}")
        
        if not self.data_directory.exists():
            logger.warning(f"Data directory does not exist: {self.data_directory}")
            self.data_directory.mkdir(parents=True, exist_ok=True)
    
    def list_available_symbols(self) -> List[str]:
        """
        Get a list of all available symbols from parquet files.
        
        Returns:
            List of symbol names (without .parquet extension)
        """
        symbols = []
        try:
            # Look for parquet files in the main directory and subdirectories
            parquet_files = list(self.data_directory.rglob("*.parquet"))
            
            for file_path in parquet_files:
                # Extract symbol name from filename or parent directory
                if file_path.parent.name != "parquet_files":
                    # Symbol is in subdirectory name
                    symbol = file_path.parent.name.upper()
                else:
                    # Symbol is in filename
                    symbol = file_path.stem.upper()
                
                if symbol not in symbols:
                    symbols.append(symbol)
            
            logger.info(f"Found {len(symbols)} symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
            return sorted(symbols)
            
        except Exception as e:
            logger.error(f"Error listing symbols: {e}")
            return []
    
    def load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load data for a specific symbol.
        
        Args:
            symbol: The symbol to load data for
            
        Returns:
            DataFrame with the symbol's data, or None if not found
        """
        try:
            symbol_lower = symbol.lower()
            symbol_upper = symbol.upper()
            
            # Try different possible file locations
            possible_paths = [
                self.data_directory / f"{symbol_upper}.parquet",
                self.data_directory / f"{symbol_lower}.parquet",
                self.data_directory / symbol_lower / f"{symbol_upper}.parquet",
                self.data_directory / symbol_lower / f"{symbol_lower}.parquet",
            ]
            
            # Also search for any parquet files in symbol subdirectory
            symbol_dir = self.data_directory / symbol_lower
            if symbol_dir.exists():
                possible_paths.extend(symbol_dir.glob("*.parquet"))
            
            df_list = []
            for file_path in possible_paths:
                if file_path.exists():
                    try:
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            df['symbol'] = symbol_upper
                            df['source_file'] = str(file_path.name)
                            df_list.append(df)
                            logger.debug(f"Loaded {len(df)} rows from {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
            
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)
                # Remove duplicates based on timestamp
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                    combined_df = combined_df.sort_values('timestamp')
                
                logger.info(f"Loaded {len(combined_df)} total rows for {symbol}")
                return combined_df
            else:
                logger.warning(f"No data found for symbol {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def load_all_data(self, symbols: Optional[List[str]] = None, min_rows: int = 100) -> pd.DataFrame:
        """
        Load data for all symbols or a specific list of symbols.
        
        Args:
            symbols: List of symbols to load. If None, loads all available symbols.
            min_rows: Minimum number of rows required per symbol to include it.
            
        Returns:
            Combined DataFrame with all symbol data
        """
        if symbols is None:
            symbols = self.list_available_symbols()
        
        all_data = []
        successful_symbols = []
        
        for symbol in symbols:
            df = self.load_symbol_data(symbol)
            if df is not None and len(df) >= min_rows:
                all_data.append(df)
                successful_symbols.append(symbol)
            elif df is not None:
                logger.warning(f"Symbol {symbol} has only {len(df)} rows, skipping (min: {min_rows})")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully loaded data for {len(successful_symbols)} symbols")
            logger.info(f"Total combined dataset: {len(combined_df)} rows")
            
            # Log some basic statistics
            if 'timestamp' in combined_df.columns:
                date_range = f"{combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}"
                logger.info(f"Date range: {date_range}")
            
            return combined_df
        else:
            logger.error("No valid data found for any symbols")
            return pd.DataFrame()
    
    def get_data_summary(self) -> dict:
        """
        Get a summary of available data.
        
        Returns:
            Dictionary with data statistics
        """
        symbols = self.list_available_symbols()
        summary = {
            'total_symbols': len(symbols),
            'symbols': symbols,
            'symbol_details': {}
        }
        
        for symbol in symbols[:20]:  # Limit to first 20 for performance
            df = self.load_symbol_data(symbol)
            if df is not None:
                summary['symbol_details'][symbol] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'date_range': {
                        'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else 'Unknown',
                        'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else 'Unknown'
                    }
                }
        
        return summary

# Legacy compatibility function for existing helper.py files
def fetch_parquet_data_from_local():
    """
    Enhanced data loader that tries local storage first, then Google Drive as fallback.
    """
    logger.info("Loading data from local storage...")
    
    loader = LocalDataLoader()
    df = loader.load_all_data(min_rows=50)
    
    # If no local data or insufficient data, try Google Drive
    if df.empty or len(df) < 1000:
        logger.warning(f"Insufficient local data ({len(df)} rows). Attempting Google Drive fallback...")
        try:
            from src.config import USE_GOOGLE_DRIVE
            if USE_GOOGLE_DRIVE:
                df_drive = _fetch_data_from_drive()
                if not df_drive.empty:
                    logger.info(f"✅ Successfully loaded {len(df_drive)} rows from Google Drive")
                    return df_drive
                else:
                    logger.warning("⚠️ No data found in Google Drive either")
            else:
                logger.info("Google Drive integration disabled in config")
        except Exception as e:
            logger.warning(f"⚠️ Google Drive fallback failed: {e}")
    
    if df.empty:
        raise ValueError("No valid parquet files found in local storage or Google Drive.")
    
    logger.info(f"Successfully loaded {len(df)} rows from local storage")
    return df

def _fetch_data_from_drive():
    """
    Fetch data from Google Drive as fallback.
    Downloads recent files to local cache and loads them.
    """
    try:
        from src.drive_manager import EnhancedDriveManager
        from pathlib import Path
        import tempfile
        
        logger.info("🔄 Attempting to download data from Google Drive...")
        
        # Initialize drive manager
        drive_manager = EnhancedDriveManager()
        if not drive_manager.authenticated:
            logger.warning("❌ Google Drive not authenticated")
            return pd.DataFrame()
        
        # Create temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Try to download recent parquet files
            # This is a simplified approach - in a full implementation,
            # you'd list files in the Drive and download them
            logger.info("📥 Downloading recent parquet files from Google Drive...")
            
            # For now, return empty DataFrame as the drive manager doesn't have
            # a direct download method exposed yet
            logger.warning("⚠️ Google Drive download not fully implemented yet")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Failed to fetch data from Google Drive: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the data loader
    loader = LocalDataLoader()
    
    print("=== Data Summary ===")
    summary = loader.get_data_summary()
    print(f"Total symbols: {summary['total_symbols']}")
    
    if summary['symbol_details']:
        print("\nSymbol details:")
        for symbol, details in summary['symbol_details'].items():
            print(f"  {symbol}: {details['rows']} rows, {details['date_range']['start']} to {details['date_range']['end']}")
    
    # Test loading all data
    print("\n=== Loading All Data ===")
    df = loader.load_all_data(min_rows=10)
    if not df.empty:
        print(f"Loaded {len(df)} total rows")
        print(f"Columns: {list(df.columns)}")
        print(f"Symbols: {df['symbol'].nunique() if 'symbol' in df.columns else 'Unknown'}")

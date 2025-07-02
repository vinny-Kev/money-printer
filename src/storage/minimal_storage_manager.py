"""
Minimal Storage Manager for Production Training
Simple local-only storage for when full storage manager has issues
"""
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

class MinimalStorageManager:
    """Minimal storage manager for production training - local files only"""
    
    def __init__(self, local_backup_dir: str):
        self.local_backup_dir = Path(local_backup_dir)
        self.memory_data = {}
        
    def load_combined_data(self, min_rows: int = 50) -> pd.DataFrame:
        """Load all parquet files from local directory"""
        if not self.local_backup_dir.exists():
            logger.error(f"Local backup directory does not exist: {self.local_backup_dir}")
            return pd.DataFrame()
            
        parquet_files = list(self.local_backup_dir.glob("*.parquet"))
        if not parquet_files:
            logger.error(f"No parquet files found in {self.local_backup_dir}")
            return pd.DataFrame()
            
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        combined_data = []
        symbols_loaded = []
        
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                if len(df) >= min_rows:
                    combined_data.append(df)
                    symbols_loaded.append(file_path.stem)
                    logger.info(f"Loaded {len(df)} rows from {file_path.name}")
                else:
                    logger.warning(f"Skipping {file_path.name}: only {len(df)} rows (min: {min_rows})")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                
        if not combined_data:
            logger.error("No valid data files loaded")
            return pd.DataFrame()
            
        result = pd.concat(combined_data, ignore_index=True)
        logger.info(f"Combined dataset: {len(result)} rows from {len(symbols_loaded)} symbols")
        
        return result
        
    def save_model_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Save model metadata to local file"""
        try:
            metadata_file = self.local_backup_dir.parent / "models" / "ensemble" / "metadata.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
            logger.info(f"Metadata saved to {metadata_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False

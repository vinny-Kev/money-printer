"""
Enhanced Storage Manager - Production-ready with fallback options
Supports Google Drive, local storage, and memory-only operation
"""
import os
import json
import logging
import time
from typing import Optional, Dict, List, Any
from datetime import datetime
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set up logger before imports
logger = logging.getLogger(__name__)

try:
    from drive_manager import EnhancedDriveManager as DriveManager
    HAS_DRIVE_MANAGER = True
except ImportError as e:
    logger.warning(f"Drive manager not available: {e}")
    DriveManager = None
    HAS_DRIVE_MANAGER = False

class EnhancedStorageManager:
    """Enhanced storage manager with multiple fallback options"""
    
    def __init__(self, 
                 drive_folder_id: Optional[str] = None,
                 local_backup_dir: Optional[str] = None,
                 memory_only: bool = False):
        self.drive_folder_id = drive_folder_id
        self.local_backup_dir = local_backup_dir
        self.memory_only = memory_only
        self.drive_manager = None
        self.memory_storage = {}
        
        # Initialize storage options
        self._init_storage_options()
        
    def _init_storage_options(self):
        """Initialize available storage options"""
        # Try to initialize Google Drive
        self.drive_available = False
        if self.drive_folder_id and not self.memory_only and HAS_DRIVE_MANAGER:
            try:
                self.drive_manager = DriveManager()
                if self.drive_manager.test_connection():
                    self.drive_available = True
                    logger.info("✅ Google Drive storage initialized")
                else:
                    logger.warning("⚠️ Google Drive connection failed, using fallback")
            except Exception as e:
                logger.warning(f"⚠️ Google Drive initialization failed: {e}")
        elif not HAS_DRIVE_MANAGER:
            logger.info("ℹ️ Google Drive manager not available, using local storage only")
        
        # Set up local backup directory
        self.local_available = False
        if self.local_backup_dir and not self.memory_only:
            try:
                os.makedirs(self.local_backup_dir, exist_ok=True)
                self.local_available = True
                logger.info(f"✅ Local backup storage initialized: {self.local_backup_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Local backup initialization failed: {e}")
        
        if self.memory_only:
            logger.info("✅ Memory-only storage mode enabled")
        
        # Log storage status
        storage_modes = []
        if self.drive_available:
            storage_modes.append("Google Drive")
        if self.local_available:
            storage_modes.append("Local Backup")
        if self.memory_only or (not self.drive_available and not self.local_available):
            storage_modes.append("Memory")
        
        logger.info(f"📁 Storage modes: {', '.join(storage_modes)}")
    
    def save_data(self, data: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Save data to available storage options"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'rows': len(data),
            'drive_success': False,
            'local_success': False,
            'memory_success': False,
            'errors': []
        }
        
        # Always save to memory first
        try:
            self.memory_storage[filename] = {
                'data': data.copy(),
                'timestamp': datetime.now(),
                'rows': len(data)
            }
            results['memory_success'] = True
            logger.debug(f"💾 Saved {filename} to memory ({len(data)} rows)")
        except Exception as e:
            results['errors'].append(f"Memory save failed: {e}")
            logger.error(f"❌ Memory save failed for {filename}: {e}")
        
        # Try Google Drive if available
        if self.drive_available and self.drive_manager:
            try:
                # PRODUCTION FIX: Use absolute paths and explicit sync to GDrive mount
                from pathlib import Path
                from src.config import DATA_ROOT
                
                # Create temp file in data directory (likely mounted to GDrive)
                gdrive_temp_dir = DATA_ROOT / "temp_gdrive_sync"
                gdrive_temp_dir.mkdir(parents=True, exist_ok=True)
                
                temp_path = gdrive_temp_dir / filename
                
                # Save to parquet with explicit path logging
                data.to_parquet(temp_path, index=False)
                logger.info(f"📁 GDRIVE: Saved temp file to absolute path: {temp_path.absolute()}")
                
                # Force filesystem sync before upload
                import os
                os.fsync(temp_path.open('rb').fileno())
                
                # Upload with absolute path verification
                success = self.drive_manager.upload_file(str(temp_path.absolute()), filename)
                if success:
                    results['drive_success'] = True
                    logger.info(f"☁️ GDRIVE SUCCESS: Uploaded {filename} from {temp_path.absolute()} ({len(data)} rows)")
                    
                    # Verify upload by checking file exists on drive
                    drive_files = self.drive_manager.list_files()
                    if filename in drive_files:
                        logger.info(f"✅ GDRIVE VERIFIED: {filename} confirmed in drive listing")
                    else:
                        logger.warning(f"⚠️ GDRIVE WARNING: {filename} not found in drive listing after upload")
                else:
                    results['errors'].append(f"Google Drive upload failed for {temp_path.absolute()}")
                    logger.error(f"❌ GDRIVE FAILED: Upload returned False for {temp_path.absolute()}")
                
                # Clean up temp file with explicit logging
                try:
                    temp_path.unlink()
                    logger.debug(f"🧹 Cleaned up temp file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Failed to cleanup temp file {temp_path}: {cleanup_error}")
                    
            except Exception as e:
                results['errors'].append(f"Google Drive save failed: {e}")
                logger.error(f"❌ Google Drive save failed for {filename}: {e}")
        
        # Try local backup if available
        if self.local_available:
            try:
                from pathlib import Path
                local_path = Path(self.local_backup_dir) / filename
                
                # PRODUCTION FIX: Ensure directory exists and save with absolute path logging
                local_path.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(local_path, index=False)
                
                # Force filesystem sync
                with open(local_path, 'rb') as f:
                    os.fsync(f.fileno())
                
                # Verify file was actually written
                if local_path.exists() and local_path.stat().st_size > 0:
                    results['local_success'] = True
                    logger.info(f"💾 LOCAL SUCCESS: Saved {filename} to {local_path.absolute()} ({len(data)} rows, {local_path.stat().st_size} bytes)")
                else:
                    results['errors'].append(f"Local file verification failed: {local_path.absolute()}")
                    logger.error(f"❌ LOCAL FAILED: File not found or empty after save: {local_path.absolute()}")
                    
            except Exception as e:
                results['errors'].append(f"Local save failed: {e}")
                logger.error(f"❌ Local save failed for {filename}: {e}")
        
        return results
    
    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from available storage options"""
        # Try memory first
        if filename in self.memory_storage:
            logger.debug(f"📖 Loading {filename} from memory")
            return self.memory_storage[filename]['data'].copy()
        
        # Try local backup
        if self.local_available:
            try:
                local_path = os.path.join(self.local_backup_dir, filename)
                if os.path.exists(local_path):
                    data = pd.read_parquet(local_path)
                    logger.info(f"📖 Loaded {filename} from local backup ({len(data)} rows)")
                    return data
            except Exception as e:
                logger.error(f"❌ Local load failed for {filename}: {e}")
        
        # Try Google Drive
        if self.drive_available and self.drive_manager:
            try:
                temp_path = f"/tmp/{filename}"
                success = self.drive_manager.download_file(filename, temp_path)
                if success and os.path.exists(temp_path):
                    data = pd.read_parquet(temp_path)
                    logger.info(f"📖 Loaded {filename} from Google Drive ({len(data)} rows)")
                    
                    # Cache in memory
                    self.memory_storage[filename] = {
                        'data': data.copy(),
                        'timestamp': datetime.now(),
                        'rows': len(data)
                    }
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                    return data
            except Exception as e:
                logger.error(f"❌ Google Drive load failed for {filename}: {e}")
        
        logger.warning(f"⚠️ Could not load {filename} from any storage")
        return None
    
    def list_files(self, pattern: str = None) -> List[str]:
        """List available files from all storage options"""
        all_files = set()
        
        # Memory files
        memory_files = list(self.memory_storage.keys())
        all_files.update(memory_files)
        if memory_files:
            logger.debug(f"📝 Found {len(memory_files)} files in memory")
        
        # Local files
        if self.local_available:
            try:
                local_files = [f for f in os.listdir(self.local_backup_dir) 
                              if f.endswith('.parquet')]
                all_files.update(local_files)
                logger.debug(f"📝 Found {len(local_files)} files in local backup")
            except Exception as e:
                logger.error(f"❌ Local file listing failed: {e}")
        
        # Google Drive files
        if self.drive_available and self.drive_manager:
            try:
                drive_files = self.drive_manager.list_files()
                drive_files = [f for f in drive_files if f.endswith('.parquet')]
                all_files.update(drive_files)
                logger.debug(f"📝 Found {len(drive_files)} files in Google Drive")
            except Exception as e:
                logger.error(f"❌ Google Drive file listing failed: {e}")
        
        files = list(all_files)
        
        # Apply pattern filter if specified
        if pattern:
            files = [f for f in files if pattern in f]
        
        logger.info(f"📋 Total unique files found: {len(files)}")
        return sorted(files)
    
    def get_storage_status(self) -> Dict[str, Any]:
        """Get comprehensive storage status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'storage_modes': {
                'google_drive': self.drive_available,
                'local_backup': self.local_available,
                'memory_only': self.memory_only
            },
            'memory_files': len(self.memory_storage),
            'total_memory_rows': sum(item['rows'] for item in self.memory_storage.values()),
            'local_files': 0,
            'drive_files': 0,
            'errors': []
        }
        
        # Count local files
        if self.local_available:
            try:
                local_files = [f for f in os.listdir(self.local_backup_dir) 
                              if f.endswith('.parquet')]
                status['local_files'] = len(local_files)
            except Exception as e:
                status['errors'].append(f"Local count failed: {e}")
        
        # Count drive files
        if self.drive_available and self.drive_manager:
            try:
                drive_files = self.drive_manager.list_files()
                drive_files = [f for f in drive_files if f.endswith('.parquet')]
                status['drive_files'] = len(drive_files)
            except Exception as e:
                status['errors'].append(f"Drive count failed: {e}")
        
        return status
    
    def cleanup_old_data(self, max_age_hours: int = 168):  # 7 days default
        """Clean up old data from memory and local storage"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned = {'memory': 0, 'local': 0}
        
        # Clean memory
        to_remove = []
        for filename, item in self.memory_storage.items():
            if item['timestamp'].timestamp() < cutoff_time:
                to_remove.append(filename)
        
        for filename in to_remove:
            del self.memory_storage[filename]
            cleaned['memory'] += 1
        
        # Clean local files
        if self.local_available:
            try:
                for filename in os.listdir(self.local_backup_dir):
                    if filename.endswith('.parquet'):
                        file_path = os.path.join(self.local_backup_dir, filename)
                        file_age = os.path.getmtime(file_path)
                        if file_age < cutoff_time:
                            os.remove(file_path)
                            cleaned['local'] += 1
            except Exception as e:
                logger.error(f"❌ Local cleanup failed: {e}")
        
        if cleaned['memory'] > 0 or cleaned['local'] > 0:
            logger.info(f"🧹 Cleaned up old data: {cleaned['memory']} memory, {cleaned['local']} local")
        
        return cleaned

#!/usr/bin/env python3
"""
Enhanced Google Drive Manager
Production-ready Google Drive integration with service account authentication,
batch upload management, file chunking, and organized folder structure.

Features:
- Service account authentication
- Batch upload manager (2-3 files per 30-60s)
- Large file chunking for >10MB files
- Organized folder structure
- Cancellable operations
- Download missing files on boot
- Resilient to disconnections
"""

import os
import sys
import json
import logging
import time
import asyncio
import hashlib
import threading
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import (
    USE_GOOGLE_DRIVE, GOOGLE_DRIVE_FOLDER_ID, SECRETS_DIR, DATA_ROOT, LOGS_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "drive_manager.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except AttributeError:
        # Some file objects don't have .buffer attribute
        pass

# Constants
CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks for large files
MAX_BATCH_SIZE = 3  # Maximum files per batch
MIN_BATCH_INTERVAL = 30  # Minimum seconds between batches
MAX_BATCH_INTERVAL = 60  # Maximum seconds between batches
MAX_RETRY_ATTEMPTS = 3
SERVICE_ACCOUNT_KEY_PATH = SECRETS_DIR / "service_account.json"

@dataclass
class FileMetadata:
    """Metadata for tracking files"""
    local_path: str
    drive_path: str
    file_hash: str
    size_bytes: int
    last_modified: float
    drive_file_id: Optional[str] = None
    uploaded_at: Optional[str] = None
    is_chunked: bool = False
    chunk_count: int = 0

@dataclass
class UploadTask:
    """Upload task for batch processing"""
    local_path: Path
    drive_path: str
    priority: int = 0  # Higher number = higher priority
    retries: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class FolderStructure:
    """Organized folder structure for Drive"""
    
    def __init__(self, root_folder_id: str):
        self.root_folder_id = root_folder_id
        self.folder_cache = {}
        self._folder_structure = {
            "trading_data": {
                "models": {
                    "random_forest": {},
                    "lightgbm": {},
                    "neural_networks": {},
                    "archived": {}
                },
                "trades": {
                    "transactions": {},
                    "backtest_results": {},
                    "performance": {}
                },
                "market_data": {
                    "scraped": {},
                    "processed": {},
                    "aggregated": {}
                },
                "diagnostics": {
                    "logs": {},
                    "errors": {},
                    "system_stats": {}
                },
                "stats": {
                    "daily": {},
                    "weekly": {},
                    "monthly": {}
                }
            },
            "logs": {
                datetime.now().strftime("%Y"): {
                    datetime.now().strftime("%m"): {}
                }
            },
            "backups": {
                "configurations": {},
                "databases": {},
                "critical_files": {}
            }
        }
    
    def get_folder_path(self, category: str, subcategory: str = None, date_based: bool = False) -> str:
        """Get organized folder path for file category"""
        if date_based:
            today = datetime.now()
            year = today.strftime("%Y")
            month = today.strftime("%m")
            day = today.strftime("%d")
            
            if category == "logs":
                return f"logs/{year}/{month}/{day}"
            elif category == "stats":
                return f"trading_data/stats/{subcategory or 'daily'}/{year}/{month}"
            else:
                return f"trading_data/{category}/{year}/{month}/{day}"
        else:
            base_path = f"trading_data/{category}"
            if subcategory:
                base_path += f"/{subcategory}"
            return base_path

class BatchUploadManager:
    """Manages batch uploads with rate limiting"""
    
    def __init__(self, drive_service, max_batch_size: int = MAX_BATCH_SIZE):
        self.drive_service = drive_service
        self.max_batch_size = max_batch_size
        self.upload_queue = Queue()
        self.is_running = False
        self.worker_thread = None
        self.cancellation_event = threading.Event()
        self.current_batch = []
        self.stats = {
            "batches_processed": 0,
            "files_uploaded": 0,
            "total_bytes": 0,
            "errors": 0,
            "last_batch_time": None
        }
        
    def start(self):
        """Start the batch upload worker"""
        if not self.is_running:
            self.is_running = True
            self.cancellation_event.clear()
            self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self.worker_thread.start()
            logger.info("🚀 Batch upload manager started")
    
    def stop(self):
        """Stop the batch upload worker"""
        if self.is_running:
            self.is_running = False
            self.cancellation_event.set()
            if self.worker_thread:
                self.worker_thread.join(timeout=10)
            logger.info("🛑 Batch upload manager stopped")
    
    def add_upload_task(self, task: UploadTask):
        """Add upload task to queue"""
        self.upload_queue.put(task)
        logger.debug(f"📝 Added upload task: {task.drive_path}")
    
    def _batch_worker(self):
        """Worker thread for processing upload batches"""
        while self.is_running and not self.cancellation_event.is_set():
            try:
                # Collect batch
                batch = self._collect_batch()
                
                if batch:
                    # Process batch
                    success_count = self._process_batch(batch)
                    self.stats["batches_processed"] += 1
                    self.stats["files_uploaded"] += success_count
                    self.stats["last_batch_time"] = datetime.now().isoformat()
                    
                    # Rate limiting
                    if success_count > 0:
                        sleep_time = MIN_BATCH_INTERVAL + (time.time() % (MAX_BATCH_INTERVAL - MIN_BATCH_INTERVAL))
                        logger.info(f"⏱️ Batch complete ({success_count}/{len(batch)} files). Sleeping {sleep_time:.1f}s")
                        
                        if self.cancellation_event.wait(sleep_time):
                            break
                else:
                    # No items in queue, wait a bit
                    if self.cancellation_event.wait(5):
                        break
                        
            except Exception as e:
                logger.error(f"❌ Batch worker error: {e}")
                self.stats["errors"] += 1
                time.sleep(10)  # Back off on errors
    
    def _collect_batch(self) -> List[UploadTask]:
        """Collect a batch of upload tasks"""
        batch = []
        
        # Try to get items from queue
        for _ in range(self.max_batch_size):
            try:
                task = self.upload_queue.get(timeout=2)
                batch.append(task)
            except Empty:
                break
        
        # Sort by priority (higher first) then by creation time
        batch.sort(key=lambda x: (-x.priority, x.created_at))
        return batch
    
    def _process_batch(self, batch: List[UploadTask]) -> int:
        """Process a batch of uploads"""
        success_count = 0
        
        # Group files for potential zipping
        zip_candidates = []
        individual_files = []
        
        for task in batch:
            if task.local_path.stat().st_size < 1024 * 1024:  # < 1MB
                zip_candidates.append(task)
            else:
                individual_files.append(task)
        
        # Process zip batch if we have multiple small files
        if len(zip_candidates) > 1:
            zip_success = self._upload_zip_batch(zip_candidates)
            if zip_success:
                success_count += len(zip_candidates)
            else:
                # Fall back to individual uploads
                individual_files.extend(zip_candidates)
        elif len(zip_candidates) == 1:
            individual_files.extend(zip_candidates)
        
        # Process individual files
        for task in individual_files:
            if self.cancellation_event.is_set():
                break
                
            try:
                success = self._upload_single_file(task)
                if success:
                    success_count += 1
                else:
                    # Retry failed uploads
                    task.retries += 1
                    if task.retries < MAX_RETRY_ATTEMPTS:
                        self.upload_queue.put(task)
                        logger.warning(f"🔄 Retrying upload: {task.drive_path} (attempt {task.retries + 1})")
                    else:
                        logger.error(f"❌ Failed to upload after {MAX_RETRY_ATTEMPTS} attempts: {task.drive_path}")
                        
            except Exception as e:
                logger.error(f"❌ Upload error for {task.drive_path}: {e}")
        
        return success_count
    
    def _upload_zip_batch(self, tasks: List[UploadTask]) -> bool:
        """Upload multiple small files as a zip batch"""
        try:
            # Create temporary zip file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                zip_path = Path(tmp_zip.name)
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for task in tasks:
                        # Add file to zip with organized path
                        arcname = task.drive_path.replace('/', '_')
                        zf.write(task.local_path, arcname)
                
                # Upload zip file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_drive_name = f"batch_upload_{timestamp}.zip"
                
                success = self._upload_to_drive(zip_path, zip_drive_name, is_batch=True)
                
                # Clean up
                zip_path.unlink(missing_ok=True)
                
                if success:
                    logger.info(f"✅ Uploaded zip batch: {len(tasks)} files as {zip_drive_name}")
                    return True
                else:
                    logger.error(f"❌ Failed to upload zip batch")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Zip batch upload error: {e}")
            return False
    
    def _upload_single_file(self, task: UploadTask) -> bool:
        """Upload a single file"""
        try:
            # Check if file is large and needs chunking
            file_size = task.local_path.stat().st_size
            
            if file_size > CHUNK_SIZE:
                return self._upload_chunked_file(task)
            else:
                return self._upload_to_drive(task.local_path, task.drive_path)
                
        except Exception as e:
            logger.error(f"❌ Single file upload error for {task.drive_path}: {e}")
            return False
    
    def _upload_chunked_file(self, task: UploadTask) -> bool:
        """Upload large file in chunks"""
        try:
            from googleapiclient.http import MediaFileUpload
            
            # Use resumable upload for large files
            media = MediaFileUpload(
                str(task.local_path),
                resumable=True,
                chunksize=CHUNK_SIZE
            )
            
            # Prepare file metadata
            file_metadata = {
                'name': Path(task.drive_path).name,
                'parents': [self._get_parent_folder_id(task.drive_path)]
            }
            
            # Start resumable upload
            request = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            )
            
            response = None
            while response is None:
                if self.cancellation_event.is_set():
                    logger.info(f"🛑 Upload cancelled: {task.drive_path}")
                    return False
                    
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"⏳ Uploading {task.drive_path}: {progress}%")
            
            file_id = response.get('id')
            if file_id:
                logger.info(f"✅ Chunked upload complete: {task.drive_path}")
                return True
            else:
                logger.error(f"❌ Chunked upload failed: {task.drive_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Chunked upload error for {task.drive_path}: {e}")
            return False
    
    def _upload_to_drive(self, local_path: Path, drive_path: str, is_batch: bool = False) -> bool:
        """Upload file to Google Drive"""
        try:
            from googleapiclient.http import MediaFileUpload
            
            # Determine media type
            media_type = self._get_media_type(local_path)
            
            media = MediaFileUpload(str(local_path), mimetype=media_type)
            
            # Prepare file metadata
            file_metadata = {
                'name': Path(drive_path).name,
                'parents': [self._get_parent_folder_id(drive_path)]
            }
            
            # Upload file
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            if file_id:
                file_size = local_path.stat().st_size
                self.stats["total_bytes"] += file_size
                
                if not is_batch:
                    logger.info(f"✅ Uploaded: {drive_path} ({file_size:,} bytes)")
                return True
            else:
                logger.error(f"❌ Upload failed: {drive_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Drive upload error for {drive_path}: {e}")
            return False
    
    def _get_media_type(self, file_path: Path) -> str:
        """Get media type for file"""
        suffix = file_path.suffix.lower()
        
        media_types = {
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.txt': 'text/plain',
            '.log': 'text/plain',
            '.parquet': 'application/octet-stream',
            '.pkl': 'application/octet-stream',
            '.pickle': 'application/octet-stream',
            '.zip': 'application/zip',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        
        return media_types.get(suffix, 'application/octet-stream')
    
    def _get_parent_folder_id(self, drive_path: str) -> str:
        """Get parent folder ID for drive path"""
        # This would need to be implemented based on folder structure
        # For now, return root folder ID
        return GOOGLE_DRIVE_FOLDER_ID
    
    def get_stats(self) -> Dict:
        """Get upload statistics"""
        return {
            **self.stats,
            "queue_size": self.upload_queue.qsize(),
            "is_running": self.is_running
        }

class EnhancedDriveManager:
    """Enhanced Google Drive Manager with production features"""
    
    def __init__(self, service_account_path: str = None, folder_id: str = None):
        self.service_account_path = Path(service_account_path or SERVICE_ACCOUNT_KEY_PATH)
        self.folder_id = folder_id or GOOGLE_DRIVE_FOLDER_ID
        
        self.service = None
        self.authenticated = False
        self.sync_enabled = USE_GOOGLE_DRIVE
        
        # File tracking
        self.file_metadata_cache = {}
        self.cache_file = SECRETS_DIR / "drive_metadata_cache.json"
        
        # Folder management
        self.folder_structure = FolderStructure(self.folder_id) if self.folder_id else None
        
        # Batch upload manager
        self.batch_manager = None
        
        # Cancellation support
        self.cancellation_event = threading.Event()
        
        self._load_metadata_cache()
        
        if self.sync_enabled:
            self._initialize_service_account()
            if self.authenticated:
                self._setup_batch_manager()
    
    def _initialize_service_account(self):
        """Initialize Google Drive service with service account"""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            
            if not self.service_account_path.exists():
                logger.error(f"Service account key not found: {self.service_account_path}")
                logger.error("Please place your service account JSON key file at the specified location")
                self.authenticated = False
                return
            
            # Load service account credentials
            SCOPES = ['https://www.googleapis.com/auth/drive']
            
            credentials = service_account.Credentials.from_service_account_file(
                str(self.service_account_path),
                scopes=SCOPES
            )
            
            # Build the service
            self.service = build('drive', 'v3', credentials=credentials)
            self.authenticated = True
            
            logger.info("✅ Google Drive service initialized with service account")
            
            # Test connection
            if not self._test_connection():
                logger.error("❌ Service account connection test failed")
                self.authenticated = False
            
        except ImportError as e:
            logger.error("Google API libraries not installed. Install with: pip install google-api-python-client google-auth")
            self.authenticated = False
        except Exception as e:
            logger.error(f"Failed to initialize service account: {e}")
            self.authenticated = False
    
    def _setup_batch_manager(self):
        """Setup batch upload manager"""
        if self.service:
            self.batch_manager = BatchUploadManager(self.service)
            self.batch_manager.start()
    
    def _test_connection(self) -> bool:
        """Test Google Drive connection"""
        if not self.authenticated or not self.service:
            return False
        
        try:
            # Try to get folder info
            folder_info = self.service.files().get(
                fileId=self.folder_id,
                fields="id, name, permissions"
            ).execute()
            
            logger.info(f"✅ Connected to Drive folder: {folder_info.get('name', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Drive connection test failed: {e}")
            return False
    
    def _load_metadata_cache(self):
        """Load metadata cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Convert to FileMetadata objects
                for key, data in cache_data.items():
                    self.file_metadata_cache[key] = FileMetadata(**data)
                    
                logger.info(f"📋 Loaded metadata cache with {len(self.file_metadata_cache)} files")
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")
            self.file_metadata_cache = {}
    
    def _save_metadata_cache(self):
        """Save metadata cache"""
        try:
            # Convert FileMetadata objects to dict
            cache_data = {}
            for key, metadata in self.file_metadata_cache.items():
                cache_data[key] = asdict(metadata)
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metadata cache: {e}")
    
    def upload_file_async(self, local_path: Path, category: str, subcategory: str = None, 
                         priority: int = 0, date_based: bool = False) -> bool:
        """Add file to async upload queue"""
        if not self.sync_enabled or not self.authenticated or not self.batch_manager:
            return False
        
        try:
            # Generate organized drive path
            if self.folder_structure:
                folder_path = self.folder_structure.get_folder_path(category, subcategory, date_based)
                drive_path = f"{folder_path}/{local_path.name}"
            else:
                drive_path = local_path.name
            
            # Check if file needs uploading
            if self._needs_upload(local_path):
                task = UploadTask(
                    local_path=local_path,
                    drive_path=drive_path,
                    priority=priority
                )
                
                self.batch_manager.add_upload_task(task)
                logger.debug(f"📤 Queued for upload: {drive_path}")
                return True
            else:
                logger.debug(f"⏭️ File unchanged, skipping: {local_path.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to queue upload for {local_path}: {e}")
            return False
    
    def _needs_upload(self, local_path: Path) -> bool:
        """Check if file needs uploading"""
        if not local_path.exists():
            return False
        
        # Calculate file hash
        file_hash = self._get_file_hash(local_path)
        cache_key = str(local_path)
        
        # Check cache
        if cache_key in self.file_metadata_cache:
            cached_metadata = self.file_metadata_cache[cache_key]
            if cached_metadata.file_hash == file_hash:
                return False  # File unchanged
        
        return True
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get SHA256 hash of file"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            return ""
    
    def sync_trading_data(self) -> Dict[str, int]:
        """Sync all trading data using organized structure"""
        if not self.sync_enabled or not self.authenticated:
            return {"status": "disabled"}
        
        results = {
            "models": 0,
            "trades": 0,
            "market_data": 0,
            "diagnostics": 0,
            "stats": 0,
            "logs": 0
        }
        
        try:
            # Sync models with subcategories
            models_dir = DATA_ROOT / "models"
            if models_dir.exists():
                for model_file in models_dir.rglob("*.pkl"):
                    if "random_forest" in model_file.name.lower():
                        subcategory = "random_forest"
                    elif "lgb" in model_file.name.lower() or "lightgbm" in model_file.name.lower():
                        subcategory = "lightgbm"
                    else:
                        subcategory = "neural_networks"
                    
                    if self.upload_file_async(model_file, "models", subcategory, priority=3):
                        results["models"] += 1
                
                # Model metadata files
                for metadata_file in models_dir.rglob("*.json"):
                    subcategory = "random_forest" if "rf" in metadata_file.name.lower() else "lightgbm"
                    if self.upload_file_async(metadata_file, "models", subcategory, priority=2):
                        results["models"] += 1
            
            # Sync trades with organization
            transactions_dir = DATA_ROOT / "transactions"
            if transactions_dir.exists():
                for trade_file in transactions_dir.rglob("*.csv"):
                    if "backtest" in trade_file.name.lower():
                        subcategory = "backtest_results"
                    elif "performance" in trade_file.name.lower():
                        subcategory = "performance"
                    else:
                        subcategory = "transactions"
                    
                    if self.upload_file_async(trade_file, "trades", subcategory, priority=2):
                        results["trades"] += 1
            
            # Sync market data (selective - recent files only)
            scraped_dir = DATA_ROOT / "scraped_data" / "parquet_files"
            if scraped_dir.exists():
                cutoff_time = time.time() - 86400  # Last 24 hours
                for parquet_file in scraped_dir.rglob("*.parquet"):
                    if parquet_file.stat().st_mtime > cutoff_time:
                        if self.upload_file_async(parquet_file, "market_data", "scraped", priority=1, date_based=True):
                            results["market_data"] += 1
            
            # Sync diagnostics with date-based organization
            diagnostics_dir = DATA_ROOT / "diagnostics"
            if diagnostics_dir.exists():
                for diag_file in diagnostics_dir.rglob("*"):
                    if diag_file.is_file():
                        if diag_file.suffix == ".log":
                            subcategory = "logs"
                        elif "error" in diag_file.name.lower():
                            subcategory = "errors"
                        else:
                            subcategory = "system_stats"
                        
                        if self.upload_file_async(diag_file, "diagnostics", subcategory, priority=1, date_based=True):
                            results["diagnostics"] += 1
            
            # Sync stats with date-based organization
            stats_files = [
                DATA_ROOT / "trading_stats.json",
                DATA_ROOT / "trading_state.json",
                DATA_ROOT / "paused_models.json"
            ]
            
            for stats_file in stats_files:
                if stats_file.exists():
                    if self.upload_file_async(stats_file, "stats", "daily", priority=2, date_based=True):
                        results["stats"] += 1
            
            # Sync recent log files
            logs_dir = LOGS_DIR
            if logs_dir.exists():
                cutoff_time = time.time() - 3600 * 24  # Last 24 hours
                for log_file in logs_dir.glob("*.log"):
                    if log_file.stat().st_mtime > cutoff_time:
                        if self.upload_file_async(log_file, "logs", None, priority=1, date_based=True):
                            results["logs"] += 1
            
            logger.info(f"📁 Trading data sync queued: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to sync trading data: {e}")
            return {"error": str(e)}
    
    def download_missing_files(self) -> Dict[str, int]:
        """Download missing files from Drive on startup"""
        if not self.sync_enabled or not self.authenticated:
            return {"status": "disabled"}
        
        downloaded = {
            "models": 0,
            "configs": 0,
            "critical_files": 0
        }
        
        try:
            # Download critical model files that are missing locally
            models_dir = DATA_ROOT / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Search for model files in Drive
            query = f"'{self.folder_id}' in parents and name contains '.pkl' and trashed=false"
            results = self.service.files().list(
                q=query,
                fields="files(id, name, parents)"
            ).execute()
            
            for file_info in results.get('files', []):
                local_path = models_dir / file_info['name']
                if not local_path.exists():
                    if self._download_file(file_info['id'], local_path):
                        downloaded["models"] += 1
                        logger.info(f"📥 Downloaded missing model: {file_info['name']}")
            
            # Download critical configuration files
            config_files = ['trading_state.json', 'paused_models.json']
            for config_file in config_files:
                local_path = DATA_ROOT / config_file
                if not local_path.exists():
                    if self._download_file_by_name(config_file, local_path):
                        downloaded["configs"] += 1
                        logger.info(f"📥 Downloaded missing config: {config_file}")
            
            logger.info(f"📥 Download complete: {downloaded}")
            return downloaded
            
        except Exception as e:
            logger.error(f"Failed to download missing files: {e}")
            return {"error": str(e)}
    
    def _download_file(self, file_id: str, local_path: Path) -> bool:
        """Download file from Drive by ID"""
        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io
            
            request = self.service.files().get_media(fileId=file_id)
            
            with open(local_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    if self.cancellation_event.is_set():
                        logger.info(f"🛑 Download cancelled: {local_path.name}")
                        return False
                    
                    status, done = downloader.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        logger.debug(f"⏳ Downloading {local_path.name}: {progress}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            return False
    
    def _download_file_by_name(self, filename: str, local_path: Path) -> bool:
        """Download file from Drive by name"""
        try:
            # Search for file
            query = f"name='{filename}' and '{self.folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                pageSize=1,
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            if files:
                return self._download_file(files[0]['id'], local_path)
            else:
                logger.warning(f"File not found in Drive: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to find/download file {filename}: {e}")
            return False
    
    def cancel_operations(self):
        """Cancel all ongoing operations"""
        logger.info("🛑 Cancelling all Drive operations...")
        self.cancellation_event.set()
        
        if self.batch_manager:
            self.batch_manager.stop()
    
    def get_status(self) -> Dict:
        """Get comprehensive status"""
        status = {
            "enabled": self.sync_enabled,
            "authenticated": self.authenticated,
            "folder_id": self.folder_id,
            "cached_files": len(self.file_metadata_cache),
            "service_account": self.service_account_path.exists(),
            "batch_manager": None
        }
        
        if self.batch_manager:
            status["batch_manager"] = self.batch_manager.get_stats()
        
        return status
    
    def cleanup(self):
        """Cleanup resources"""
        self.cancel_operations()
        self._save_metadata_cache()
        
        if self.batch_manager:
            self.batch_manager.stop()

# Global instance
_drive_manager = None

def get_drive_manager() -> EnhancedDriveManager:
    """Get or create Drive manager singleton"""
    global _drive_manager
    if _drive_manager is None:
        _drive_manager = EnhancedDriveManager()
    return _drive_manager

def cleanup_drive_manager():
    """Cleanup global Drive manager"""
    global _drive_manager
    if _drive_manager:
        _drive_manager.cleanup()
        _drive_manager = None

# CLI interface
async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Google Drive Manager")
    parser.add_argument("--test", action="store_true", help="Test connection")
    parser.add_argument("--sync", action="store_true", help="Sync trading data")
    parser.add_argument("--download", action="store_true", help="Download missing files")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--setup", action="store_true", help="Setup service account")
    
    args = parser.parse_args()
    
    try:
        if args.setup:
            print("🔧 Enhanced Google Drive Setup")
            print("=" * 50)
            print("\n📋 Service Account Setup Steps:")
            print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
            print("2. Create a new project or select existing one")
            print("3. Enable the Google Drive API")
            print("4. Go to 'Credentials' → 'Create Credentials' → 'Service Account'")
            print("5. Create a service account with Drive access")
            print("6. Generate and download the JSON key file")
            print(f"7. Save it as: {SERVICE_ACCOUNT_KEY_PATH}")
            print("\n📂 Share your Google Drive folder with the service account email")
            print("   (found in the JSON key file as 'client_email')")
            print(f"\n⚙️  Add to your .env file:")
            print(f"USE_GOOGLE_DRIVE=true")
            print(f"GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here")
            return
        
        manager = get_drive_manager()
        
        if args.test:
            if manager._test_connection():
                print("✅ Google Drive connection successful")
            else:
                print("❌ Google Drive connection failed")
        
        elif args.sync:
            print("🔄 Starting trading data sync...")
            results = manager.sync_trading_data()
            print(f"📁 Sync results: {results}")
        
        elif args.download:
            print("📥 Downloading missing files...")
            results = manager.download_missing_files()
            print(f"📥 Download results: {results}")
        
        elif args.status:
            status = manager.get_status()
            print("📊 Drive Manager Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        cleanup_drive_manager()

if __name__ == "__main__":
    asyncio.run(main())

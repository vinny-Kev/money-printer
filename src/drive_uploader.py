#!/usr/bin/env python3
"""
Google Drive Integration
Automatic syncing of training/trade data to Google Drive for backup and analysis.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import (
    USE_GOOGLE_DRIVE, GOOGLE_DRIVE_FOLDER_ID, DRIVE_CREDENTIALS_PATH,
    DRIVE_TOKEN_PATH, SECRETS_DIR, DATA_ROOT, LOGS_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "drive_sync.log", encoding='utf-8'),
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

class DriveUploader:
    """Google Drive file upload and sync manager"""
    
    def __init__(self, credentials_path: str = None, token_path: str = None, folder_id: str = None):
        self.credentials_path = Path(credentials_path or DRIVE_CREDENTIALS_PATH)
        self.token_path = Path(token_path or DRIVE_TOKEN_PATH)
        self.folder_id = folder_id or GOOGLE_DRIVE_FOLDER_ID
        
        self.service = None
        self.authenticated = False
        self.sync_enabled = USE_GOOGLE_DRIVE
        
        # Track uploaded files to avoid duplicates
        self.uploaded_files_cache = {}
        self.cache_file = SECRETS_DIR / "drive_upload_cache.json"
        self._load_upload_cache()
        
        if self.sync_enabled:
            self._initialize_drive_service()
    
    def _load_upload_cache(self):
        """Load cache of uploaded files"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.uploaded_files_cache = json.load(f)
                logger.info(f"Loaded upload cache with {len(self.uploaded_files_cache)} files")
        except Exception as e:
            logger.warning(f"Failed to load upload cache: {e}")
            self.uploaded_files_cache = {}
    
    def _save_upload_cache(self):
        """Save cache of uploaded files"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.uploaded_files_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save upload cache: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get SHA256 hash of file for change detection"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            return ""
    
    def _initialize_drive_service(self):
        """Initialize Google Drive API service"""
        try:
            # Import Google API libraries
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            
            creds = None
            
            # Load existing token
            if self.token_path.exists():
                creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not self.credentials_path.exists():
                        logger.error(f"Google credentials not found at {self.credentials_path}")
                        logger.error("Please download credentials.json from Google Cloud Console")
                        self.authenticated = False
                        return
                        
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path), SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
            
            self.service = build('drive', 'v3', credentials=creds)
            self.authenticated = True
            logger.info("✅ Google Drive service initialized successfully")
            
        except ImportError as e:
            logger.error("Google API libraries not installed. Install with: pip install google-api-python-client google-auth google-auth-oauthlib")
            self.authenticated = False
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            self.authenticated = False
    
    def test_connection(self) -> bool:
        """Test Google Drive connection"""
        if not self.authenticated or not self.service:
            return False
            
        try:
            # Try to list files in the folder
            results = self.service.files().list(
                q=f"'{self.folder_id}' in parents and trashed=false",
                pageSize=1,
                fields="nextPageToken, files(id, name)"
            ).execute()
            
            logger.info("✅ Google Drive connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Google Drive connection test failed: {e}")
            return False
    
    def upload_file(self, local_path: Path, drive_filename: str = None) -> Optional[str]:
        """
        Upload a file to Google Drive
        Returns the file ID if successful, None otherwise
        """
        if not self.sync_enabled or not self.authenticated:
            return None
            
        try:
            from googleapiclient.http import MediaFileUpload
            
            if not local_path.exists():
                logger.error(f"File not found: {local_path}")
                return None
            
            # Use filename if not specified
            filename = drive_filename or local_path.name
            
            # Check if file has changed since last upload
            file_hash = self._get_file_hash(local_path)
            cache_key = str(local_path)
            
            if cache_key in self.uploaded_files_cache:
                cached_info = self.uploaded_files_cache[cache_key]
                if cached_info.get('hash') == file_hash:
                    logger.debug(f"File unchanged, skipping upload: {filename}")
                    return cached_info.get('file_id')
            
            # Prepare file metadata
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id] if self.folder_id else []
            }
            
            # Determine media type
            media_type = None
            if local_path.suffix.lower() in ['.csv', '.json']:
                media_type = 'text/plain'
            elif local_path.suffix.lower() == '.parquet':
                media_type = 'application/octet-stream'
            elif local_path.suffix.lower() in ['.pkl', '.pickle']:
                media_type = 'application/octet-stream'
            
            media = MediaFileUpload(str(local_path), mimetype=media_type, resumable=True)
            
            # Check if file already exists in Drive
            existing_file_id = self._find_existing_file(filename)
            
            if existing_file_id:
                # Update existing file
                file = self.service.files().update(
                    fileId=existing_file_id,
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                logger.info(f"📁 Updated file in Google Drive: {filename}")
            else:
                # Create new file
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                logger.info(f"📁 Uploaded new file to Google Drive: {filename}")
            
            file_id = file.get('id')
            
            # Update cache
            self.uploaded_files_cache[cache_key] = {
                'file_id': file_id,
                'hash': file_hash,
                'uploaded_at': datetime.utcnow().isoformat(),
                'filename': filename
            }
            self._save_upload_cache()
            
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to upload file {local_path}: {e}")
            return None
    
    def _find_existing_file(self, filename: str) -> Optional[str]:
        """Find existing file in Drive folder"""
        try:
            query = f"name='{filename}'"
            if self.folder_id:
                query += f" and '{self.folder_id}' in parents"
            query += " and trashed=false"
            
            results = self.service.files().list(
                q=query,
                pageSize=1,
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            if files:
                return files[0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Failed to search for existing file: {e}")
            return None
    
    def sync_directory(self, local_dir: Path, pattern: str = "*", recursive: bool = True) -> int:
        """
        Sync files from a directory to Google Drive
        Returns number of files synced
        """
        if not self.sync_enabled or not self.authenticated:
            return 0
            
        synced_count = 0
        
        try:
            if recursive:
                files = list(local_dir.rglob(pattern))
            else:
                files = list(local_dir.glob(pattern))
            
            for file_path in files:
                if file_path.is_file():
                    # Create relative path for Drive filename
                    rel_path = file_path.relative_to(local_dir)
                    drive_filename = str(rel_path).replace(os.sep, '_')
                    
                    if self.upload_file(file_path, drive_filename):
                        synced_count += 1
            
            logger.info(f"📁 Synced {synced_count} files from {local_dir}")
            return synced_count
            
        except Exception as e:
            logger.error(f"Failed to sync directory {local_dir}: {e}")
            return 0
    
    def sync_trading_data(self) -> Dict[str, int]:
        """Sync all trading-related data to Google Drive"""
        if not self.sync_enabled:
            return {"status": "disabled"}
            
        results = {
            "models": 0,
            "trades": 0,
            "diagnostics": 0,
            "scraped_data": 0,
            "stats": 0
        }
        
        try:
            # Sync model files
            models_dir = DATA_ROOT / "models"
            if models_dir.exists():
                results["models"] = self.sync_directory(models_dir, "*.pkl")
                results["models"] += self.sync_directory(models_dir, "*.json")
            
            # Sync trade data
            transactions_dir = DATA_ROOT / "transactions"
            if transactions_dir.exists():
                results["trades"] = self.sync_directory(transactions_dir, "*.csv")
            
            # Sync diagnostics
            diagnostics_dir = DATA_ROOT / "diagnostics"
            if diagnostics_dir.exists():
                results["diagnostics"] = self.sync_directory(diagnostics_dir, "*.json")
                results["diagnostics"] += self.sync_directory(diagnostics_dir, "*.png")
            
            # Sync scraped data (selective - only recent files)
            scraped_dir = DATA_ROOT / "scraped_data" / "parquet_files"
            if scraped_dir.exists():
                # Only sync files modified in the last 24 hours
                cutoff_time = time.time() - 86400  # 24 hours ago
                for file_path in scraped_dir.rglob("*.parquet"):
                    if file_path.stat().st_mtime > cutoff_time:
                        if self.upload_file(file_path):
                            results["scraped_data"] += 1
            
            # Sync stats files
            stats_files = [
                DATA_ROOT / "trading_stats.json",
                DATA_ROOT / "trading_state.json",
                DATA_ROOT / "paused_models.json"
            ]
            
            for stats_file in stats_files:
                if stats_file.exists():
                    if self.upload_file(stats_file):
                        results["stats"] += 1
            
            logger.info(f"📁 Trading data sync complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to sync trading data: {e}")
            results["error"] = str(e)
            return results
    
    def get_sync_status(self) -> Dict:
        """Get current sync status"""
        return {
            "enabled": self.sync_enabled,
            "authenticated": self.authenticated,
            "folder_id": self.folder_id,
            "cached_files": len(self.uploaded_files_cache),
            "last_sync": self._get_last_sync_time()
        }
    
    def _get_last_sync_time(self) -> Optional[str]:
        """Get timestamp of last sync"""
        if not self.uploaded_files_cache:
            return None
            
        timestamps = [
            info.get('uploaded_at') for info in self.uploaded_files_cache.values()
            if info.get('uploaded_at')
        ]
        
        if timestamps:
            return max(timestamps)
        return None

# Global instance
_drive_uploader = None

def get_drive_uploader() -> DriveUploader:
    """Get or create Drive uploader singleton"""
    global _drive_uploader
    if _drive_uploader is None:
        _drive_uploader = DriveUploader()
    return _drive_uploader

async def main():
    """CLI entry point for Drive uploader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Google Drive Uploader")
    parser.add_argument("--test", action="store_true", help="Test Google Drive connection")
    parser.add_argument("--sync", action="store_true", help="Sync all trading data")
    parser.add_argument("--upload", type=str, help="Upload specific file")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    
    args = parser.parse_args()
    
    try:
        uploader = get_drive_uploader()
        
        if args.test:
            if uploader.test_connection():
                print("✅ Google Drive connection successful")
            else:
                print("❌ Google Drive connection failed")
                
        elif args.sync:
            print("Starting trading data sync...")
            results = uploader.sync_trading_data()
            print(f"Sync results: {results}")
            
        elif args.upload:
            file_path = Path(args.upload)
            if file_path.exists():
                file_id = uploader.upload_file(file_path)
                if file_id:
                    print(f"✅ Uploaded {file_path.name} (ID: {file_id})")
                else:
                    print(f"❌ Failed to upload {file_path.name}")
            else:
                print(f"❌ File not found: {file_path}")
                
        elif args.status:
            status = uploader.get_sync_status()
            print(f"Sync Status:")
            print(f"  Enabled: {status['enabled']}")
            print(f"  Authenticated: {status['authenticated']}")
            print(f"  Folder ID: {status['folder_id']}")
            print(f"  Cached Files: {status['cached_files']}")
            print(f"  Last Sync: {status['last_sync']}")
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Drive uploader stopped by user")
    except Exception as e:
        logger.error(f"Drive uploader error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

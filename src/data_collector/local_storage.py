import os
import json
import time
import logging
import random
import pandas as pd
from datetime import datetime
from io import BytesIO
import shutil

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Local storage configuration
LOCAL_DATA_DIR = os.path.join(os.getcwd(), "data", "scraped_data")
LOCAL_LOGS_DIR = os.path.join(os.getcwd(), "logs")
PARQUET_DATA_DIR = os.path.join(LOCAL_DATA_DIR, "parquet_files")
DAILY_UPLOAD_LIMIT_MB = 500  # Keep for compatibility but not strictly enforced locally
MAX_STORAGE_GB = 10  # Maximum local storage limit in GB

# Ensure directories exist
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)
os.makedirs(PARQUET_DATA_DIR, exist_ok=True)

# Status tracking files
STATUS_FILE = os.path.join(LOCAL_DATA_DIR, "storage_status.json")
UPLOAD_TRACKER_FILE = os.path.join(LOCAL_DATA_DIR, "daily_tracker.json")

# Fun quotes for Discord alerts (keeping the humor)
peasant_quotes = [
    "My lord! Local storage is getting full!",
    "Forgive me, my lord — the disk space runs low!",
    "The bytes accumulate, sire! The disk grows heavy!",
    "We've stored much data, my lord. The hard drive groans!",
    "Sire, should we clean old files to make space?"
]

def send_discord_alert(message):
    """
    Send an alert message to a Discord channel via webhook.
    This is kept for compatibility but Discord integration is optional for local storage.
    """
    # Import here to avoid circular imports
    try:
        from dotenv import load_dotenv
        import requests
        load_dotenv()
        
        DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
        if not DISCORD_WEBHOOK:
            logger.warning("⚠️ No Discord webhook set. Skipping alert.")
            return

        quote = random.choice(peasant_quotes)
        payload = {
             "content": f"🏠 **{quote}**\n\n{message}"
        }
        response = requests.post(DISCORD_WEBHOOK, json=payload)
        if response.status_code == 204:
            logger.info("✅ Discord alert sent.")
        else:
            logger.warning(f"❌ Failed to send Discord alert: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Exception while sending Discord alert: {e}")

def get_directory_size(path):
    """
    Calculate the total size of a directory in bytes.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                # Handle cases where file might be deleted during iteration
                pass
    return total_size

def check_storage_space():
    """
    Check local storage usage and return status.
    """
    try:
        # Calculate storage used by our data directory
        storage_used = get_directory_size(LOCAL_DATA_DIR)
        storage_limit = MAX_STORAGE_GB * 1024 * 1024 * 1024  # Convert GB to bytes
        
        storage_used_gb = storage_used / (1024 ** 3)
        is_full = storage_used >= storage_limit
        
        logger.info(f"📁 Local storage used: {storage_used_gb:.2f} GB / {MAX_STORAGE_GB} GB")
        
        # Update status file
        status = {
            "storage_full": is_full,
            "storage_used_bytes": storage_used,
            "storage_limit_bytes": storage_limit,
            "storage_used_gb": storage_used_gb,
            "last_checked": datetime.utcnow().isoformat()
        }
        
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)
        
        if is_full:
            send_discord_alert(f"🚨 **Local Storage Full**: {storage_used_gb:.2f} GB / {MAX_STORAGE_GB} GB")
        elif storage_used_gb > 0.9 * MAX_STORAGE_GB:
            send_discord_alert(f"⚠️ **Local Storage High**: {storage_used_gb:.2f} GB / {MAX_STORAGE_GB} GB")
        
        return storage_used, storage_limit
        
    except Exception as e:
        logger.error(f"❌ Error checking local storage: {e}")
        return 0, MAX_STORAGE_GB * 1024 * 1024 * 1024

def save_parquet_file(data, filename, symbol=None):
    """
    Save parquet data to local storage and upload to Google Drive if enabled.
    :param data: DataFrame or bytes data to save
    :param filename: Name of the file to save
    :param symbol: Optional symbol name for organization
    """
    try:
        # Create symbol-specific directory if provided
        if symbol:
            symbol_dir = os.path.join(PARQUET_DATA_DIR, symbol.lower())
            os.makedirs(symbol_dir, exist_ok=True)
            filepath = os.path.join(symbol_dir, filename)
        else:
            filepath = os.path.join(PARQUET_DATA_DIR, filename)
        
        # Save the data locally first
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath, index=False, compression="snappy")
        elif isinstance(data, bytes):
            with open(filepath, "wb") as f:
                f.write(data)
        else:
            # Assume it's a file-like object
            if hasattr(data, 'read'):
                with open(filepath, "wb") as f:
                    f.write(data.read())
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        
        file_size = os.path.getsize(filepath)
        logger.info(f"✅ Saved parquet file locally: {filename} ({file_size / (1024 * 1024):.2f} MB)")
        
        # Check storage after saving
        check_storage_space()
        
        # Upload to Google Drive if enabled
        try:
            # Import here to avoid circular imports
            from src.config import USE_GOOGLE_DRIVE
            if USE_GOOGLE_DRIVE:
                logger.info(f"🔄 Attempting Google Drive upload for {filename}...")
                from src.drive_manager import EnhancedDriveManager
                
                # Initialize drive manager
                drive_manager = EnhancedDriveManager()
                
                # Use sync upload since we're in a sync context
                import asyncio
                try:
                    # Create event loop if one doesn't exist
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Upload the file
                upload_result = loop.run_until_complete(
                    drive_manager.upload_file_async(filepath, filename)
                )
                
                if upload_result:
                    logger.info(f"☁️ Successfully uploaded {filename} to Google Drive")
                else:
                    logger.warning(f"⚠️ Failed to upload {filename} to Google Drive")
            else:
                logger.debug("Google Drive integration disabled")
        except Exception as drive_error:
            logger.warning(f"⚠️ Google Drive upload failed for {filename}: {drive_error}")
            # Don't fail the entire operation if Drive upload fails
        
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Failed to save parquet file {filename}: {e}")
        return None

def load_parquet_file(filename, symbol=None):
    """
    Load parquet data from local storage.
    """
    try:
        if symbol:
            filepath = os.path.join(PARQUET_DATA_DIR, symbol.lower(), filename)
        else:
            filepath = os.path.join(PARQUET_DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            return None
            
        return pd.read_parquet(filepath)
        
    except Exception as e:
        logger.error(f"❌ Failed to load parquet file {filename}: {e}")
        return None

def list_parquet_files(symbol=None):
    """
    List all parquet files in local storage.
    """
    try:
        if symbol:
            search_dir = os.path.join(PARQUET_DATA_DIR, symbol.lower())
        else:
            search_dir = PARQUET_DATA_DIR
            
        if not os.path.exists(search_dir):
            return []
            
        files = []
        for root, dirs, filenames in os.walk(search_dir):
            for filename in filenames:
                if filename.endswith('.parquet'):
                    full_path = os.path.join(root, filename)
                    # Get relative path from PARQUET_DATA_DIR
                    rel_path = os.path.relpath(full_path, PARQUET_DATA_DIR)
                    files.append({
                        'filename': filename,
                        'path': full_path,
                        'full_path': full_path,  # Add this for compatibility
                        'relative_path': rel_path,
                        'size': os.path.getsize(full_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(full_path))
                    })
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)
        
    except Exception as e:
        logger.error(f"❌ Failed to list parquet files: {e}")
        return []

def cleanup_old_files(days_old=30, max_files_per_symbol=100):
    """
    Clean up old parquet files to free space.
    """
    try:
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        files_deleted = 0
        space_freed = 0
        
        for root, dirs, filenames in os.walk(PARQUET_DATA_DIR):
            symbol_files = []
            for filename in filenames:
                if filename.endswith('.parquet'):
                    filepath = os.path.join(root, filename)
                    file_stat = os.stat(filepath)
                    symbol_files.append({
                        'path': filepath,
                        'modified': file_stat.st_mtime,
                        'size': file_stat.st_size
                    })
            
            # Sort by modification time (oldest first)
            symbol_files.sort(key=lambda x: x['modified'])
            
            # Delete old files or excess files
            for i, file_info in enumerate(symbol_files):
                should_delete = False
                
                # Delete if too old
                if file_info['modified'] < cutoff_date:
                    should_delete = True
                    reason = f"older than {days_old} days"
                
                # Delete if we have too many files for this symbol
                elif i < len(symbol_files) - max_files_per_symbol:
                    should_delete = True
                    reason = f"keeping only latest {max_files_per_symbol} files"
                
                if should_delete:
                    try:
                        os.remove(file_info['path'])
                        files_deleted += 1
                        space_freed += file_info['size']
                        logger.info(f"🗑️ Deleted {file_info['path']} ({reason})")
                    except Exception as e:
                        logger.error(f"❌ Failed to delete {file_info['path']}: {e}")
        
        if files_deleted > 0:
            space_freed_mb = space_freed / (1024 * 1024)
            logger.info(f"🧹 Cleanup complete: deleted {files_deleted} files, freed {space_freed_mb:.2f} MB")
            send_discord_alert(f"🧹 **Cleanup Complete**: Deleted {files_deleted} files, freed {space_freed_mb:.2f} MB")
        
        return files_deleted, space_freed
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        return 0, 0

def check_and_pause_if_storage_full():
    """
    Check if storage is full and pause if necessary.
    This replaces the bucket full check.
    """
    try:
        storage_used, storage_limit = check_storage_space()
        if storage_used >= storage_limit:
            logger.warning("⚠️ Local storage is full. Running cleanup...")
            send_discord_alert("⚠️ **Storage Full**: Running automatic cleanup...")
            
            # Try cleanup first
            files_deleted, space_freed = cleanup_old_files()
            
            # Check again after cleanup
            storage_used, storage_limit = check_storage_space()
            if storage_used >= storage_limit:
                logger.warning("⚠️ Storage still full after cleanup. Pausing operations...")
                return False
            else:
                logger.info("✅ Storage has available space after cleanup. Continuing...")
                return True
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking storage status: {e}")
        return True  # Continue operation on error

def local_log(message, level="INFO"):
    """
    Save a local log message.
    """
    try:
        log_path = os.path.join(LOCAL_LOGS_DIR, f"local_storage_{datetime.utcnow().date()}.log")
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(f"{datetime.utcnow()} [{level}] {message}\n")
    except Exception as e:
        logger.error(f"❌ Failed to write local log: {e}")

# Legacy compatibility functions (to ease migration)
def upload_file(data, remote_filename=None, bucket_type="data", is_binary=True):
    """
    Legacy compatibility function - now saves locally instead of uploading.
    """
    if not remote_filename:
        raise ValueError("filename must be provided.")
    
    # Extract symbol from filename if possible
    symbol = None
    if remote_filename.endswith('.parquet'):
        symbol = os.path.splitext(remote_filename)[0]
    
    return save_parquet_file(data, remote_filename, symbol)

def check_bucket_storage():
    """
    Legacy compatibility function - returns local storage info.
    """
    return check_storage_space()

def check_and_pause_if_bucket_full():
    """
    Legacy compatibility function.
    """
    return check_and_pause_if_storage_full()

if __name__ == "__main__":
    # Test the local storage system
    print("Testing local storage system...")
    
    # Check storage
    storage_used, storage_limit = check_storage_space()
    print(f"Storage: {storage_used / (1024**3):.2f} GB / {storage_limit / (1024**3):.2f} GB")
    
    # List existing files
    files = list_parquet_files()
    print(f"Found {len(files)} parquet files")
    
    if files:
        print("Recent files:")
        for file_info in files[:5]:
            print(f"  {file_info['filename']} - {file_info['size'] / 1024:.1f} KB - {file_info['modified']}")

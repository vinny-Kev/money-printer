"""
Test to verify if files are actually being uploaded to Google Drive
"""
import os
import sys
import logging
import tempfile
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def verify_upload_success():
    """Verify that files are actually being uploaded successfully"""
    
    logger.info("🧪 Verifying Upload Success...")
    
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
        'price': [100, 101, 102, 103, 104]
    })
    
    # Create temporary file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"VERIFY_UPLOAD_{timestamp}.parquet"
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    # Rename to proper filename
    temp_path = Path(temp_file_path)
    final_temp_path = temp_path.parent / filename
    temp_path.rename(final_temp_path)
    
    # Save test data
    test_data.to_parquet(final_temp_path, index=False)
    logger.info(f"📁 Created test file: {final_temp_path} ({final_temp_path.stat().st_size} bytes)")
    
    try:
        # Initialize drive manager
        from src.drive_manager import EnhancedDriveManager
        drive_manager = EnhancedDriveManager()
        
        if not drive_manager.authenticated:
            logger.error("❌ Google Drive not authenticated")
            return
        
        # Step 1: Count files before upload
        logger.info("📊 Step 1: Counting files before upload...")
        files_before = drive_manager.list_files_in_folder()
        parquet_before = [f for f in files_before if f.get('name', '').endswith('.parquet')]
        logger.info(f"   Files before: {len(files_before)} total, {len(parquet_before)} parquet")
        
        # Step 2: Queue upload
        logger.info("📤 Step 2: Queueing upload...")
        result = drive_manager.upload_file_async(
            final_temp_path,
            "verify_test",
            "upload_verify",
            priority=1,
            date_based=False
        )
        logger.info(f"   Upload queued: {result}")
        
        # Step 3: Wait for background processing
        logger.info("⏱️ Step 3: Waiting for background processing...")
        for i in range(12):  # Wait up to 60 seconds
            time.sleep(5)
            queue_size = drive_manager.batch_manager.upload_queue.qsize()
            stats = drive_manager.batch_manager.stats
            logger.info(f"   Wait {(i+1)*5}s: Queue={queue_size}, Stats={stats}")
            
            if queue_size == 0 and stats['files_uploaded'] > 0:
                logger.info("   ✅ Upload appears to be complete!")
                break
        else:
            logger.warning("   ⚠️ Timeout waiting for upload completion")
        
        # Step 4: Count files after upload
        logger.info("📊 Step 4: Counting files after upload...")
        files_after = drive_manager.list_files_in_folder()
        parquet_after = [f for f in files_after if f.get('name', '').endswith('.parquet')]
        logger.info(f"   Files after: {len(files_after)} total, {len(parquet_after)} parquet")
        
        # Step 5: Look for our specific file
        logger.info("🔍 Step 5: Looking for our uploaded file...")
        our_file = None
        for file in files_after:
            if filename in file.get('name', ''):
                our_file = file
                break
        
        if our_file:
            logger.info(f"   ✅ SUCCESS! Found our file: {our_file['name']}")
            logger.info(f"      Size: {our_file.get('size', 'Unknown')} bytes")
            logger.info(f"      ID: {our_file.get('id', 'Unknown')}")
            return True
        else:
            logger.warning(f"   ❌ Could not find our file: {filename}")
            if parquet_after:
                logger.info("   Recent parquet files in Drive:")
                for file in parquet_after[-3:]:
                    name = file.get('name', 'Unknown')
                    size = file.get('size', 'Unknown')
                    modified = file.get('modifiedTime', 'Unknown')
                    logger.info(f"      {name} ({size} bytes, {modified})")
            return False
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test file (with retry in case it's locked)
        logger.info("🗑️ Cleaning up test file...")
        for attempt in range(3):
            try:
                if final_temp_path.exists():
                    final_temp_path.unlink()
                    logger.info("   ✅ File cleaned up successfully")
                break
            except PermissionError:
                logger.warning(f"   ⚠️ File locked, retrying in 2s... (attempt {attempt + 1})")
                time.sleep(2)
        else:
            logger.warning(f"   ⚠️ Could not clean up file: {final_temp_path}")

if __name__ == "__main__":
    success = verify_upload_success()
    if success:
        print("\n🎉 VERIFICATION SUCCESSFUL: Files are being uploaded to Google Drive!")
    else:
        print("\n❌ VERIFICATION FAILED: Upload system needs investigation")

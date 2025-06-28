"""
Simple test to debug the batch upload queue issue
"""
import os
import sys
import logging
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def debug_batch_upload():
    """Debug the batch upload system step by step"""
    
    logger.info("🧪 Debugging Batch Upload System...")
    
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
        'price': [100, 101, 102, 103, 104]
    })
    
    # Create temporary file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"DEBUG_UPLOAD_{timestamp}.parquet"
    
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
        
        logger.info("✅ Drive manager authenticated")
        
        # Debug step 1: Check queue before adding
        logger.info(f"🔍 Queue size before upload: {drive_manager.batch_manager.upload_queue.qsize()}")
        
        # Debug step 2: Add upload task
        logger.info("📝 Adding upload task...")
        result = drive_manager.upload_file_async(
            final_temp_path,
            "debug_test",
            "upload_debug",
            priority=1,
            date_based=False
        )
        logger.info(f"   Upload task added: {result}")
        
        # Debug step 3: Check queue after adding
        logger.info(f"🔍 Queue size after upload: {drive_manager.batch_manager.upload_queue.qsize()}")
        
        # Wait a moment for the worker thread to process
        import time
        time.sleep(3)
        logger.info(f"🔍 Queue size after 3s wait: {drive_manager.batch_manager.upload_queue.qsize()}")
        
        # Debug step 4: Check batch manager stats immediately
        stats = drive_manager.batch_manager.stats
        logger.info(f"📊 Batch manager stats after wait: {stats}")
        
        # Debug step 4: Check if task is in queue
        if not drive_manager.batch_manager.upload_queue.empty():
            logger.info("✅ Task found in queue")
        else:
            logger.warning("⚠️ No task in queue!")
            return
        
        # Debug step 5: Try to process manually
        logger.info("🔄 Processing uploads manually...")
        
        # First, let's see what's in the queue
        temp_tasks = []
        while not drive_manager.batch_manager.upload_queue.empty():
            try:
                task = drive_manager.batch_manager.upload_queue.get_nowait()
                temp_tasks.append(task)
                logger.info(f"   Found task: {task.local_path} -> {task.drive_path}")
            except Exception as e:
                logger.error(f"   Error getting task: {e}")
                break
        
        # Put tasks back
        for task in temp_tasks:
            drive_manager.batch_manager.upload_queue.put(task)
        
        # Now try to process
        processed = drive_manager.batch_manager.process_pending_uploads()
        logger.info(f"   Files processed: {processed}")
        
        # Debug step 6: Check files in Drive
        logger.info("📁 Checking files in Drive...")
        files_after = drive_manager.list_files_in_folder()
        parquet_files_after = [f for f in files_after if f.get('name', '').endswith('.parquet')]
        logger.info(f"   Parquet files in Drive: {len(parquet_files_after)}")
        
        if parquet_files_after:
            logger.info("   Recent files:")
            for file in parquet_files_after[-3:]:
                logger.info(f"      {file.get('name', 'Unknown')} ({file.get('size', 'Unknown')} bytes)")
        
        # Debug step 7: Check stats
        stats = drive_manager.batch_manager.get_stats() if hasattr(drive_manager.batch_manager, 'get_stats') else drive_manager.batch_manager.stats
        logger.info(f"📊 Batch manager stats: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test file
        if final_temp_path.exists():
            final_temp_path.unlink()
            logger.info(f"🗑️ Cleaned up test file")

if __name__ == "__main__":
    debug_batch_upload()

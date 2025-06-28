"""
Test the batch upload system to ensure files are being uploaded to Google Drive
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_batch_upload_system():
    """Test that the batch upload system is working correctly"""
    
    logger.info("🧪 Testing Batch Upload System...")
    
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
        'open': [100 + i for i in range(10)],
        'high': [105 + i for i in range(10)],
        'low': [95 + i for i in range(10)],
        'close': [102 + i for i in range(10)],
        'volume': [1000 + i*100 for i in range(10)]
    })
    
    # Create temporary file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"TEST_BATCH_UPLOAD_{timestamp}.parquet"
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    # Rename to proper filename
    temp_path = Path(temp_file_path)
    final_temp_path = temp_path.parent / filename
    temp_path.rename(final_temp_path)
    
    # Save test data
    test_data.to_parquet(final_temp_path, index=False)
    logger.info(f"📁 Created test file: {final_temp_path}")
    
    try:
        # Initialize drive manager
        from src.drive_manager import EnhancedDriveManager
        drive_manager = EnhancedDriveManager()
        
        if not drive_manager.authenticated:
            logger.error("❌ Google Drive not authenticated")
            return
        
        # Test 1: Check current files in Drive
        logger.info("\n📋 Test 1: Current files in Google Drive")
        files_before = drive_manager.list_files_in_folder()
        parquet_files_before = [f for f in files_before if f.get('name', '').endswith('.parquet')]
        logger.info(f"   Files before upload: {len(parquet_files_before)}")
        
        # Test 2: Upload using batch system
        logger.info("\n📋 Test 2: Uploading via batch system")
        result = drive_manager.upload_file_async(
            final_temp_path,
            "test_data",
            "batch_test",
            priority=1,
            date_based=True
        )
        logger.info(f"   Batch upload queued: {result}")
        
        # Test 3: Process uploads immediately
        logger.info("\n📋 Test 3: Processing batch uploads immediately")
        processed = drive_manager.batch_manager.process_pending_uploads()
        logger.info(f"   Files processed: {processed}")
        
        # Test 4: Check files after upload
        logger.info("\n📋 Test 4: Files after upload")
        files_after = drive_manager.list_files_in_folder()
        parquet_files_after = [f for f in files_after if f.get('name', '').endswith('.parquet')]
        logger.info(f"   Files after upload: {len(parquet_files_after)}")
        
        # Look for our specific file
        our_file = None
        for file in files_after:
            if filename in file.get('name', ''):
                our_file = file
                break
        
        if our_file:
            logger.info(f"   ✅ Found our test file: {our_file['name']}")
            logger.info(f"      Size: {our_file.get('size', 'Unknown')} bytes")
            logger.info(f"      Modified: {our_file.get('modifiedTime', 'Unknown')}")
        else:
            logger.warning(f"   ⚠️ Could not find our test file: {filename}")
            if parquet_files_after:
                logger.info("   Recent parquet files:")
                for file in parquet_files_after[-5:]:  # Show last 5
                    logger.info(f"      {file.get('name', 'Unknown')}")
        
        # Test 5: Direct upload comparison
        logger.info("\n📋 Test 5: Testing direct upload for comparison")
        try:
            direct_filename = f"TEST_DIRECT_UPLOAD_{timestamp}.parquet"
            direct_temp_path = final_temp_path.parent / direct_filename
            test_data.to_parquet(direct_temp_path, index=False)
            
            # Direct upload using Google Drive API
            media = drive_manager._create_media_upload(direct_temp_path)
            file_metadata = {
                'name': direct_filename,
                'parents': [drive_manager.folder_id]
            }
            
            uploaded_file = drive_manager.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,size,modifiedTime'
            ).execute()
            
            logger.info(f"   ✅ Direct upload successful: {uploaded_file['name']}")
            logger.info(f"      File ID: {uploaded_file['id']}")
            
            # Clean up direct test file
            direct_temp_path.unlink()
            
        except Exception as e:
            logger.error(f"   ❌ Direct upload failed: {e}")
        
        # Test 6: Check batch manager status
        logger.info("\n📋 Test 6: Batch manager status")
        if hasattr(drive_manager, 'batch_manager'):
            queue_size = drive_manager.batch_manager.get_queue_size()
            logger.info(f"   Current queue size: {queue_size}")
            
            stats = getattr(drive_manager.batch_manager, 'stats', {})
            if stats:
                logger.info(f"   Upload stats: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test file
        if final_temp_path.exists():
            final_temp_path.unlink()
            logger.info(f"🗑️ Cleaned up test file: {final_temp_path}")

if __name__ == "__main__":
    test_batch_upload_system()

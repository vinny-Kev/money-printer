#!/usr/bin/env python3
"""
Simple Google Drive Test
Direct test of Google Drive upload and file listing functionality
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_drive_basic():
    """Test basic Google Drive operations"""
    print("🔗 Testing Google Drive Basic Operations")
    print("=" * 50)
    
    try:
        from drive_manager import EnhancedDriveManager
        
        # Initialize drive manager
        print("📋 Initializing Google Drive manager...")
        dm = EnhancedDriveManager()
        
        # Test 1: List existing files
        print("\n📁 Test 1: Listing existing files...")
        files = dm.list_files_in_folder()
        print(f"   Found {len(files)} files in Google Drive")
        
        if files:
            print("   📋 Existing files:")
            for i, file_info in enumerate(files[:5]):  # Show first 5
                name = file_info.get('name', 'Unknown')
                size = file_info.get('size', 0)
                print(f"     {i+1}. {name} ({size} bytes)")
        else:
            print("   📭 No files found")
        
        # Test 2: Create and upload a simple test file
        print("\n📤 Test 2: Creating and uploading test file...")
        
        # Create a simple test file
        test_data = {
            "test": "Google Drive Upload Test",
            "timestamp": datetime.now().isoformat(),
            "message": "Hello from Money Printer!"
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(test_data, temp_file, indent=2)
            temp_file_path = temp_file.name
        
        print(f"   💾 Created test file: {temp_file_path}")
        
        # Try direct upload using Google Drive API
        try:
            from googleapiclient.http import MediaFileUpload
            
            # Prepare file metadata
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            drive_filename = f"simple_drive_test_{timestamp}.json"
            
            file_metadata = {
                'name': drive_filename,
                'parents': [dm.folder_id]
            }
            
            # Upload file
            media = MediaFileUpload(temp_file_path, mimetype='application/json')
            file = dm.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            if file_id:
                print(f"   ✅ File uploaded successfully!")
                print(f"   📁 File ID: {file_id}")
                print(f"   📄 Filename: {drive_filename}")
                
                # Verify upload by listing files again
                print("\n🔍 Test 3: Verifying upload...")
                new_files = dm.list_files_in_folder()
                print(f"   📁 Total files now: {len(new_files)}")
                
                # Look for our uploaded file
                uploaded_file = None
                for f in new_files:
                    if f.get('name') == drive_filename:
                        uploaded_file = f
                        break
                
                if uploaded_file:
                    print(f"   ✅ Upload verified! File found in Drive")
                    print(f"   📄 Name: {uploaded_file.get('name')}")
                    print(f"   📊 Size: {uploaded_file.get('size')} bytes")
                    return True
                else:
                    print(f"   ❌ File not found in Drive listing")
                    return False
            else:
                print(f"   ❌ Upload failed - no file ID returned")
                return False
                
        except Exception as upload_error:
            print(f"   ❌ Upload failed: {upload_error}")
            return False
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
                print(f"   🧹 Cleaned up temp file")
            except:
                pass
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_upload():
    """Test the batch upload system"""
    print("\n🚀 Testing Batch Upload System")
    print("=" * 50)
    
    try:
        from drive_manager import EnhancedDriveManager
        
        dm = EnhancedDriveManager()
        
        # Create a test file
        test_content = f"Batch upload test - {datetime.now()}"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        print(f"📝 Created test file: {temp_file_path}")
        
        # Try batch upload
        temp_path = Path(temp_file_path)
        result = dm.upload_file_async(temp_path, "test", "batch")
        
        print(f"📤 Batch upload queued: {result}")
        
        # Process batch immediately
        if hasattr(dm, 'batch_manager') and dm.batch_manager:
            print("⏱️ Processing batch immediately...")
            uploaded_count = dm.batch_manager.process_pending_uploads()
            print(f"📊 Processed {uploaded_count} uploads")
            
            if uploaded_count > 0:
                print("✅ Batch upload successful!")
                return True
            else:
                print("❌ No files processed")
                return False
        else:
            print("❌ No batch manager available")
            return False
            
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False
    finally:
        try:
            os.unlink(temp_file_path)
        except:
            pass

def main():
    """Run all tests"""
    print("🏁 Starting Simple Google Drive Test")
    print("📅 " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()
    
    # Check environment
    folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
    if not folder_id:
        print("❌ GOOGLE_DRIVE_FOLDER_ID not set in environment")
        return False
    
    service_key_path = "secrets/service_account.json"
    if not os.path.exists(service_key_path):
        print(f"❌ Service account key not found: {service_key_path}")
        return False
    
    print(f"✅ Environment check passed")
    print(f"📁 Drive folder ID: {folder_id}")
    print(f"🔑 Service key: {service_key_path}")
    
    # Run tests
    results = []
    
    # Test 1: Basic operations
    print("\n" + "="*60)
    basic_result = test_drive_basic()
    results.append(("Basic Drive Operations", basic_result))
    
    # Test 2: Batch upload
    print("\n" + "="*60)
    batch_result = test_batch_upload()
    results.append(("Batch Upload System", batch_result))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Google Drive is working perfectly!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()

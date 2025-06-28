#!/usr/bin/env python3
"""
Google Drive Location Test
Check where files are actually being uploaded
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_drive_locations():
    """Check different locations in Google Drive"""
    print("🔍 Checking Google Drive File Locations")
    print("=" * 50)
    
    try:
        from drive_manager import EnhancedDriveManager
        
        dm = EnhancedDriveManager()
        service = dm.service
        folder_id = dm.folder_id
        
        print(f"📁 Main folder ID: {folder_id}")
        
        # 1. Search for our test file by name
        print("\n📋 Searching for test files by name...")
        query = "name contains 'simple_drive_test'"
        results = service.files().list(
            q=query,
            fields="files(id, name, parents, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        print(f"   Found {len(files)} test files:")
        for f in files:
            print(f"     📄 {f['name']}")
            print(f"        ID: {f['id']}")
            print(f"        Parents: {f.get('parents', [])}")
            print(f"        Size: {f.get('size', 0)} bytes")
        
        # 2. Check what's in the main folder
        print(f"\n📁 Files in main folder ({folder_id}):")
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, size, modifiedTime)"
        ).execute()
        
        files = results.get('files', [])
        print(f"   Found {len(files)} files in main folder:")
        for f in files:
            print(f"     📄 {f['name']} ({f.get('size', 0)} bytes)")
        
        # 3. Check for subfolders
        print(f"\n📁 Subfolders in main folder:")
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name)"
        ).execute()
        
        folders = results.get('files', [])
        print(f"   Found {len(folders)} subfolders:")
        for folder in folders:
            print(f"     📁 {folder['name']} (ID: {folder['id']})")
            
            # Check what's in each subfolder
            subfolder_query = f"'{folder['id']}' in parents and trashed=false"
            subfolder_results = service.files().list(
                q=subfolder_query,
                fields="files(id, name, size)"
            ).execute()
            
            subfolder_files = subfolder_results.get('files', [])
            print(f"        Contains {len(subfolder_files)} files:")
            for sf in subfolder_files[:3]:  # Show first 3
                print(f"          📄 {sf['name']} ({sf.get('size', 0)} bytes)")
            if len(subfolder_files) > 3:
                print(f"          ... and {len(subfolder_files) - 3} more")
        
        # 4. Try to access the file we know was uploaded
        print(f"\n🔍 Checking specific uploaded file...")
        file_id = "1SOA0X04j8FW2FfqELYZNBeXI2zRZ-EUc"
        try:
            file_info = service.files().get(
                fileId=file_id,
                fields="id, name, parents, size, mimeType"
            ).execute()
            
            print(f"   ✅ File found!")
            print(f"     📄 Name: {file_info['name']}")
            print(f"     📁 Parents: {file_info.get('parents', [])}")
            print(f"     📊 Size: {file_info.get('size', 0)} bytes")
            
            # Check if parent matches our folder
            parents = file_info.get('parents', [])
            if folder_id in parents:
                print(f"   ✅ File is in the correct folder!")
            else:
                print(f"   ⚠️ File is in a different folder: {parents}")
        
        except Exception as e:
            print(f"   ❌ Could not access file: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_drive_locations()

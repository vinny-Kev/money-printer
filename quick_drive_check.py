"""
Quick verification that uploaded files are in Google Drive
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def check_drive_files():
    """Check files in Google Drive"""
    from drive_manager import EnhancedDriveManager
    
    dm = EnhancedDriveManager()
    files = dm.list_files_in_folder()
    parquet_files = [f for f in files if f.get('name', '').endswith('.parquet')]
    
    print(f"📁 Total files in Drive: {len(files)}")
    print(f"📄 Parquet files in Drive: {len(parquet_files)}")
    
    if parquet_files:
        print("\n📋 Recent parquet files:")
        for file in sorted(parquet_files, key=lambda x: x.get('modifiedTime', ''))[-5:]:
            name = file.get('name', 'Unknown')
            size = file.get('size', 'Unknown')
            modified = file.get('modifiedTime', 'Unknown')
            print(f"  {name} ({size} bytes, {modified})")
    
    return len(parquet_files)

if __name__ == "__main__":
    check_drive_files()

#!/usr/bin/env python3
"""
Debug script to check if scraper files are being created and saved
"""
import os
import time
from datetime import datetime
from src.config import PARQUET_DATA_DIR

def check_scraper_files():
    """Check what files the scraper has created"""
    
    print(f"🔍 Checking for scraped data files...")
    print(f"📁 Data directory: {PARQUET_DATA_DIR}")
    print(f"⏰ Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Convert to string if it's a Path object
    data_dir = str(PARQUET_DATA_DIR)
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        return
    
    # List all files and directories
    total_files = 0
    total_size = 0
    
    print(f"\n📊 Directory Contents:")
    print("-" * 60)
    
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        
        if root != data_dir:
            print(f"{indent}📂 {os.path.basename(root)}/")
        
        # Show files in this directory
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            print(f"{subindent}📄 {file}")
            print(f"{subindent}   Size: {file_size / 1024:.2f} KB")
            print(f"{subindent}   Modified: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            total_files += 1
            total_size += file_size
    
    print("-" * 60)
    print(f"📊 Summary:")
    print(f"   Total files: {total_files}")
    print(f"   Total size: {total_size / (1024 * 1024):.2f} MB")
    
    if total_files == 0:
        print(f"⚠️ No files found! Possible issues:")
        print(f"   1. Scraper hasn't run long enough (needs 60+ seconds)")
        print(f"   2. No data is being received from Binance")
        print(f"   3. Buffer is empty or save function failed")
        print(f"   4. Permission issues with file creation")
    else:
        print(f"✅ Files found! Scraper is working correctly.")
    
    # Check for recent files (created in last 5 minutes)
    recent_files = []
    five_minutes_ago = time.time() - 300
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getmtime(file_path) > five_minutes_ago:
                recent_files.append(file_path)
    
    if recent_files:
        print(f"\n🔥 Recent files (last 5 minutes): {len(recent_files)}")
        for file_path in recent_files[:10]:  # Show first 10
            rel_path = os.path.relpath(file_path, data_dir)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"   📄 {rel_path} ({file_time.strftime('%H:%M:%S')})")
    else:
        print(f"\n❌ No recent files found (last 5 minutes)")

if __name__ == "__main__":
    check_scraper_files()

#!/usr/bin/env python3
"""
Debug script to check scraper process and buffer status
"""
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import CACHE_DIR

def check_scraper_buffer():
    """Check the scraper buffer status"""
    
    print(f"🔍 Checking scraper buffer status...")
    print(f"⏰ Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for buffer cache file
    cache_file = os.path.join(CACHE_DIR, "ohlcv_buffer.json")
    print(f"📁 Buffer cache file: {cache_file}")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                buffer_data = json.load(f)
            
            print(f"✅ Buffer file found!")
            print(f"📊 Buffer contents:")
            
            total_records = 0
            for symbol, data in buffer_data.items():
                record_count = len(data) if data else 0
                total_records += record_count
                if record_count > 0:
                    print(f"   {symbol}: {record_count} records")
            
            print(f"\n📈 Total symbols with data: {len([s for s, d in buffer_data.items() if d])}")
            print(f"📈 Total records in buffer: {total_records}")
            
            if total_records == 0:
                print(f"⚠️ Buffer is empty! Possible issues:")
                print(f"   1. Scraper just started (data comes in slowly)")
                print(f"   2. WebSocket connections not working")
                print(f"   3. Binance API issues")
            else:
                print(f"🔥 Buffer has data! Should save on next cycle.")
            
        except Exception as e:
            print(f"❌ Error reading buffer file: {e}")
    else:
        print(f"❌ Buffer file not found. Scraper may not be running or just started.")
    
    # Check for running processes
    print(f"\n🔍 Checking for running Python processes...")
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'scraper' in cmdline.lower() or 'data_collector' in cmdline.lower():
                        print(f"   🐍 Found scraper process: PID {proc.info['pid']}")
                        print(f"      Command: {cmdline}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        print(f"   ⚠️ psutil not available, can't check processes")

if __name__ == "__main__":
    check_scraper_buffer()

#!/usr/bin/env python3
"""
Test script to diagnose data_scraper import issues
"""

import sys
import os

print("Testing data_scraper import...")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    print("Attempting to import data_scraper...")
    import data_scraper
    print("✅ data_scraper import successful!")
    
    print("Testing main function exists...")
    if hasattr(data_scraper, 'main'):
        print("✅ main function found")
    else:
        print("❌ main function not found")
        
    print("Testing handle_sigint function exists...")
    if hasattr(data_scraper, 'handle_sigint'):
        print("✅ handle_sigint function found")
    else:
        print("❌ handle_sigint function not found")
        
except Exception as e:
    print(f"❌ Import failed: {e}")
    print(f"Error type: {type(e)}")
    
    # Try to see what's in the file
    try:
        with open('data_scraper.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\nFirst 15 lines of data_scraper.py:")
            for i, line in enumerate(lines[:15], 1):
                print(f"{i:2d}: {line.rstrip()}")
    except Exception as read_error:
        print(f"❌ Could not read file: {read_error}")

#!/usr/bin/env python3
"""
Simple Windows-Compatible Integration Test
Tests Railway and Google Drive without Unicode console issues.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_railway_config():
    """Test Railway configuration"""
    print("Testing Railway Configuration...")
    print("-" * 40)
    
    # Check environment variables
    api_token = os.getenv("RAILWAY_API_TOKEN")
    project_id = os.getenv("RAILWAY_PROJECT_ID")
    
    if api_token and project_id:
        print("✓ Railway credentials found")
        print(f"  Project ID: {project_id[:8]}...")
        print(f"  Token: {api_token[:8]}...")
        
        # Try to import and create watchdog
        try:
            from src.railway_watchdog import RailwayWatchdog
            watchdog = RailwayWatchdog(api_token, project_id)
            print("✓ Railway watchdog created successfully")
            return True
        except Exception as e:
            print(f"✗ Railway watchdog failed: {e}")
            return False
    else:
        print("✗ Railway credentials not configured")
        print("  Please set RAILWAY_API_TOKEN and RAILWAY_PROJECT_ID in .env")
        return False

def test_drive_config():
    """Test Google Drive configuration"""
    print("\nTesting Google Drive Configuration...")
    print("-" * 40)
    
    # Check environment variables
    use_drive = os.getenv("USE_GOOGLE_DRIVE", "false").lower() == "true"
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    
    if not use_drive:
        print("○ Google Drive sync is disabled")
        return True
    
    if use_drive and folder_id:
        print("✓ Google Drive configuration found")
        print(f"  Folder ID: {folder_id[:20]}...")
        
        # Check for credentials file
        from src.config import DRIVE_CREDENTIALS_PATH, DRIVE_TOKEN_PATH
        
        if DRIVE_CREDENTIALS_PATH.exists():
            print("✓ Google credentials file found")
        else:
            print("✗ Google credentials file missing")
            print(f"  Expected at: {DRIVE_CREDENTIALS_PATH}")
            return False
            
        # Try to import and create uploader
        try:
            from src.drive_uploader import DriveUploader
            uploader = DriveUploader()
            print("✓ Drive uploader created successfully")
            
            status = uploader.get_sync_status()
            print(f"  Enabled: {status['enabled']}")
            print(f"  Authenticated: {status['authenticated']}")
            return True
        except Exception as e:
            print(f"✗ Drive uploader failed: {e}")
            return False
    else:
        print("✗ Google Drive not properly configured")
        if use_drive and not folder_id:
            print("  Please set GOOGLE_DRIVE_FOLDER_ID in .env")
        return False

def test_discord_commands():
    """Test if Discord bot can import new commands"""
    print("\nTesting Discord Bot Integration...")
    print("-" * 40)
    
    try:
        # Test if the Discord bot can import the new modules
        from src.trading_bot.discord_trader_bot import bot
        print("✓ Discord bot imports successfully")
        
        # Check if the new command functions exist
        commands = [cmd.name for cmd in bot.commands]
        new_commands = ["usage_status", "drive_status", "drive_sync"]
        
        found_commands = [cmd for cmd in new_commands if cmd in commands]
        
        if found_commands:
            print(f"✓ New commands available: {', '.join(found_commands)}")
        else:
            print("✗ New commands not found")
            
        return len(found_commands) > 0
        
    except Exception as e:
        print(f"✗ Discord bot import failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting File Structure...")
    print("-" * 40)
    
    required_files = [
        "src/railway_watchdog.py",
        "src/drive_uploader.py",
        "background_services.py",
        "setup_integrations.py",
        "test_integrations.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n✗ Missing files: {len(missing_files)}")
        return False
    else:
        print(f"\n✓ All {len(required_files)} files present")
        return True

def main():
    """Run all tests"""
    print("Railway & Google Drive Integration Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Railway Config", test_railway_config),
        ("Google Drive Config", test_drive_config),
        ("Discord Integration", test_discord_commands),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{status:>6} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\nALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Configure Railway API credentials in .env")
        print("2. Set up Google Drive integration with setup_integrations.py")
        print("3. Start background services with background_services.py")
        print("4. Use Discord commands: /usage_status, /drive_status")
    else:
        print("\nSome tests failed. Please check the configuration.")
        
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        input("\nPress Enter to exit...")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        sys.exit(1)

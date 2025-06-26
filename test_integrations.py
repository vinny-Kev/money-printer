#!/usr/bin/env python3
"""
Integration Test Script
Tests Railway usage monitoring and Google Drive integration.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import USE_GOOGLE_DRIVE
from src.railway_watchdog import get_railway_watchdog
from src.drive_uploader import get_drive_uploader

async def test_railway_integration():
    """Test Railway usage monitoring"""
    print("🚂 Testing Railway Integration...")
    print("=" * 40)
    
    try:
        watchdog = get_railway_watchdog()
        
        # Test basic usage check
        print("📊 Checking usage...")
        usage = await watchdog.check_usage_once()
        
        if usage:
            print(f"✅ Usage check successful:")
            print(f"   Current: {usage.current_hours:.2f} hours")
            print(f"   Limit: {usage.limit_hours} hours") 
            print(f"   Percentage: {usage.usage_percentage:.1f}%")
            print(f"   Remaining: {usage.remaining_hours:.2f} hours")
            print(f"   Estimated Cost: ${usage.estimated_monthly_cost:.2f}")
        else:
            print("❌ Failed to get usage data")
            
        # Test status method
        print("\n📋 Testing status method...")
        status = watchdog.get_usage_status()
        
        if "error" not in status:
            print(f"✅ Status method working:")
            print(f"   Status: {status['status']}")
            print(f"   Usage: {status['usage_percentage']:.1f}%")
        else:
            print(f"❌ Status method failed: {status['error']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Railway integration test failed: {e}")
        return False

def test_drive_integration():
    """Test Google Drive integration"""
    print("\n📁 Testing Google Drive Integration...")
    print("=" * 40)
    
    try:
        uploader = get_drive_uploader()
        
        # Test configuration
        print("⚙️ Checking configuration...")
        status = uploader.get_sync_status()
        
        print(f"   Enabled: {status['enabled']}")
        print(f"   Authenticated: {status['authenticated']}")
        print(f"   Folder ID: {status['folder_id']}")
        print(f"   Cached Files: {status['cached_files']}")
        
        if not status['enabled']:
            print("⏸️ Google Drive sync is disabled")
            return True
            
        if not status['authenticated']:
            print("❌ Google Drive not authenticated")
            return False
            
        # Test connection
        print("\n🔐 Testing connection...")
        if uploader.test_connection():
            print("✅ Google Drive connection successful")
        else:
            print("❌ Google Drive connection failed")
            return False
            
        # Test file upload with a small test file
        print("\n📤 Testing file upload...")
        test_file = Path("test_upload.json")
        
        try:
            # Create test file
            test_data = {
                "test": True,
                "timestamp": "2025-06-25T23:00:00Z",
                "message": "Test file for Drive integration"
            }
            
            with open(test_file, 'w') as f:
                json.dump(test_data, f, indent=2)
            
            # Upload test file
            file_id = uploader.upload_file(test_file, "trading_bot_test.json")
            
            if file_id:
                print(f"✅ Test file uploaded successfully (ID: {file_id})")
            else:
                print("❌ Test file upload failed")
                
            # Clean up
            test_file.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"❌ File upload test failed: {e}")
            test_file.unlink(missing_ok=True)
            return False
            
        # Test sync status
        print("\n📊 Testing sync functionality...")
        sync_results = uploader.sync_trading_data()
        
        if "error" not in sync_results:
            total_synced = sum([
                sync_results.get('models', 0),
                sync_results.get('trades', 0),
                sync_results.get('diagnostics', 0),
                sync_results.get('scraped_data', 0),
                sync_results.get('stats', 0)
            ])
            print(f"✅ Sync test completed: {total_synced} files processed")
            print(f"   Models: {sync_results.get('models', 0)}")
            print(f"   Trades: {sync_results.get('trades', 0)}")
            print(f"   Diagnostics: {sync_results.get('diagnostics', 0)}")
            print(f"   Data: {sync_results.get('scraped_data', 0)}")
            print(f"   Stats: {sync_results.get('stats', 0)}")
        else:
            print(f"❌ Sync test failed: {sync_results['error']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Google Drive integration test failed: {e}")
        return False

def test_environment_config():
    """Test environment configuration"""
    print("⚙️ Testing Environment Configuration...")
    print("=" * 40)
    
    required_vars = [
        "RAILWAY_API_TOKEN",
        "RAILWAY_PROJECT_ID"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file")
        return False
    else:
        print("✅ All required environment variables are set")
    
    # Check optional variables
    optional_vars = [
        ("USE_GOOGLE_DRIVE", os.getenv("USE_GOOGLE_DRIVE", "false")),
        ("GOOGLE_DRIVE_FOLDER_ID", os.getenv("GOOGLE_DRIVE_FOLDER_ID", "not set")),
        ("RAILWAY_MAX_USAGE_HOURS", os.getenv("RAILWAY_MAX_USAGE_HOURS", "450")),
        ("RAILWAY_WARNING_HOURS", os.getenv("RAILWAY_WARNING_HOURS", "400"))
    ]
    
    print("\n📋 Optional configuration:")
    for var_name, var_value in optional_vars:
        print(f"   {var_name}: {var_value}")
    
    return True

async def run_all_tests():
    """Run all integration tests"""
    print("🧪 Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Config", test_environment_config),
        ("Railway Integration", test_railway_integration),
        ("Google Drive Integration", test_drive_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n💡 Next steps:")
        print("1. Start background services: python background_services.py")
        print("2. Use Discord commands: /usage_status, /drive_status")
        print("3. Monitor logs for automatic sync and usage alerts")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration.")
        
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)

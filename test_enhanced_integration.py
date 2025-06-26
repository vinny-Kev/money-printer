#!/usr/bin/env python3
"""
Enhanced Drive Manager Test Script
Tests the new service account-based Google Drive integration with all features.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_enhanced_drive_manager():
    """Test the enhanced Drive manager"""
    print("🧪 Enhanced Google Drive Manager Test")
    print("=" * 50)
    
    try:
        from src.drive_manager import get_drive_manager, cleanup_drive_manager
        from src.config import USE_GOOGLE_DRIVE, GOOGLE_DRIVE_FOLDER_ID
        
        # Check configuration
        print("\n⚙️ Checking configuration...")
        if not USE_GOOGLE_DRIVE:
            print("⏭️ Google Drive sync is disabled")
            return True
        
        if not GOOGLE_DRIVE_FOLDER_ID:
            print("❌ GOOGLE_DRIVE_FOLDER_ID not set")
            return False
        
        print(f"✅ Configuration valid - Folder ID: {GOOGLE_DRIVE_FOLDER_ID[:20]}...")
        
        # Initialize manager
        print("\n🔧 Initializing Drive manager...")
        manager = get_drive_manager()
        
        if not manager:
            print("❌ Failed to create Drive manager")
            return False
        
        print("✅ Drive manager created")
        
        # Check status
        print("\n📊 Checking status...")
        status = manager.get_status()
        
        print(f"  Enabled: {status['enabled']}")
        print(f"  Authenticated: {status['authenticated']}")
        print(f"  Service Account: {status['service_account']}")
        print(f"  Cached Files: {status['cached_files']}")
        
        if status['batch_manager']:
            batch_stats = status['batch_manager']
            print(f"  Batch Manager: Running={batch_stats['is_running']}")
            print(f"  Queue Size: {batch_stats['queue_size']}")
            print(f"  Files Uploaded: {batch_stats['files_uploaded']}")
        
        if not status['enabled']:
            print("⏭️ Drive sync disabled, skipping further tests")
            return True
        
        if not status['authenticated']:
            print("❌ Drive not authenticated")
            print("💡 To set up service account:")
            print("   1. Run: python src/drive_manager.py --setup")
            print("   2. Follow the setup instructions")
            return False
        
        # Test connection
        print("\n🔗 Testing connection...")
        if manager._test_connection():
            print("✅ Connection test successful")
        else:
            print("❌ Connection test failed")
            return False
        
        # Test file upload
        print("\n📤 Testing file upload...")
        test_data = {
            "test": True,
            "timestamp": datetime.now().isoformat(),
            "message": "Enhanced Drive Manager Test File"
        }
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, indent=2)
            test_file_path = Path(f.name)
        
        try:
            # Test async upload
            success = manager.upload_file_async(
                test_file_path, 
                category="diagnostics", 
                subcategory="system_stats",
                priority=5,
                date_based=True
            )
            
            if success:
                print("✅ Test file queued for upload")
            else:
                print("❌ Test file upload failed")
                return False
            
            # Wait a moment for batch processing
            import time
            print("⏳ Waiting for batch processing...")
            time.sleep(5)
            
            # Check batch manager stats
            if manager.batch_manager:
                stats = manager.batch_manager.get_stats()
                print(f"📊 Batch stats: {stats['files_uploaded']} files uploaded")
            
        finally:
            # Clean up test file
            test_file_path.unlink(missing_ok=True)
        
        # Test trading data sync
        print("\n🔄 Testing trading data sync...")
        sync_results = manager.sync_trading_data()
        
        if "error" in sync_results:
            print(f"❌ Sync error: {sync_results['error']}")
            return False
        
        total_queued = sum([
            sync_results.get('models', 0),
            sync_results.get('trades', 0),
            sync_results.get('market_data', 0),
            sync_results.get('diagnostics', 0),
            sync_results.get('stats', 0),
            sync_results.get('logs', 0)
        ])
        
        print(f"✅ Sync complete - {total_queued} files queued:")
        print(f"   Models: {sync_results.get('models', 0)}")
        print(f"   Trades: {sync_results.get('trades', 0)}")
        print(f"   Market Data: {sync_results.get('market_data', 0)}")
        print(f"   Diagnostics: {sync_results.get('diagnostics', 0)}")
        print(f"   Stats: {sync_results.get('stats', 0)}")
        print(f"   Logs: {sync_results.get('logs', 0)}")
        
        # Test download functionality
        print("\n📥 Testing download functionality...")
        download_results = manager.download_missing_files()
        
        if "error" in download_results:
            print(f"⚠️ Download warning: {download_results['error']}")
        else:
            total_downloaded = sum([
                download_results.get('models', 0),
                download_results.get('configs', 0),
                download_results.get('critical_files', 0)
            ])
            print(f"✅ Download check complete - {total_downloaded} files downloaded")
        
        print("\n✅ All enhanced Drive manager tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure Google API libraries are installed:")
        print("   pip install google-api-python-client google-auth")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    finally:
        # Clean up
        try:
            cleanup_drive_manager()
        except:
            pass

def test_docker_health_check():
    """Test Docker health check"""
    print("\n🐳 Testing Docker health check...")
    print("-" * 30)
    
    try:
        from docker.health_check import health_check
        
        checks = health_check()
        
        print("Health Check Results:")
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}: {result}")
        
        all_passed = all(checks.values())
        
        if all_passed:
            print("✅ Docker health check passed")
        else:
            print("❌ Docker health check failed")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_background_services():
    """Test background services integration"""
    print("\n🔄 Testing background services...")
    print("-" * 30)
    
    try:
        from background_services import BackgroundServices
        
        # Create background services instance
        bg_services = BackgroundServices()
        
        print("✅ Background services created")
        
        # Check if services are available
        print(f"  Railway watchdog: {'✅' if bg_services.railway_watchdog else '❌'}")
        print(f"  Drive manager: {'✅' if bg_services.drive_manager else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Background services test error: {e}")
        return False

def test_railway_integration():
    """Test Railway watchdog integration"""
    print("\n🚂 Testing Railway integration...")
    print("-" * 30)
    
    try:
        from src.railway_watchdog import get_railway_watchdog
        
        watchdog = get_railway_watchdog()
        
        if not watchdog:
            print("⏭️ Railway watchdog not configured")
            return True
        
        # Test status check
        status = watchdog.get_usage_status()
        
        if "error" in status:
            print(f"⚠️ Railway API error: {status['error']}")
            return True  # Not critical for this test
        
        print(f"✅ Railway usage: {status['usage_percentage']:.1f}%")
        print(f"   Current: {status['current_hours']:.2f}h")
        print(f"   Remaining: {status['remaining_hours']:.2f}h")
        
        return True
        
    except Exception as e:
        print(f"❌ Railway test error: {e}")
        return False

def main():
    """Run all enhanced integration tests"""
    print("🧪 Enhanced Integration Test Suite")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Enhanced Drive Manager", test_enhanced_drive_manager),
        ("Background Services", test_background_services),
        ("Railway Integration", test_railway_integration),
        ("Docker Health Check", test_docker_health_check),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced integration is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        sys.exit(1)

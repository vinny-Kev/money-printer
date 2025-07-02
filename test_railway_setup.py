#!/usr/bin/env python3
"""
Test Railway Setup Locally
This simulates what Railway will do when deploying your app
"""
import subprocess
import sys
import os
import time
import requests
from threading import Thread

def test_dependencies():
    """Test that all required dependencies can be imported."""
    print("🔍 Testing dependencies...")
    try:
        import flask
        import requests
        import datetime
        import json
        import logging
        print("✅ Core dependencies OK")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_server_start():
    """Test that production_server.py starts successfully."""
    print("🚀 Testing server startup...")
    
    # Start server in background
    try:
        process = subprocess.Popen([
            sys.executable, "production_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server started successfully")
                print(f"✅ Health endpoint working: {response.json()}")
                return True
            else:
                print(f"❌ Health endpoint returned {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to server: {e}")
            return False
        finally:
            process.terminate()
            process.wait()
            
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        return False

def main():
    """Run all Railway deployment tests."""
    print("🧪 Testing Railway Deployment Setup...")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Server Startup", test_server_start),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary:")
    all_passed = True
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🚀 Ready for Railway deployment!")
        print("Next steps:")
        print("1. Push to GitHub: git add . && git commit -m 'Railway ready' && git push")
        print("2. Go to railway.app and deploy from your GitHub repo")
        print("3. Railway will automatically detect Python and use your Procfile")
    else:
        print("\n⚠️  Fix the failing tests before deploying to Railway")

if __name__ == "__main__":
    main()

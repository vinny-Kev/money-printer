#!/usr/bin/env python3
"""
Railway Deployment Analysis - Check what will happen to scraper and data when deployed
This analyzes the current configuration and predicts Railway behavior
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def analyze_railway_deployment():
    """Analyze what will happen when deployed to Railway"""
    print("🔍 RAILWAY DEPLOYMENT ANALYSIS")
    print("=" * 60)
    
    # 1. Check configuration paths
    print("\n📁 STORAGE CONFIGURATION ANALYSIS:")
    
    try:
        from src.config import (
            DATA_ROOT, SCRAPED_DATA_DIR, PARQUET_DATA_DIR,
            USE_GOOGLE_DRIVE, GOOGLE_DRIVE_FOLDER_ID,
            PROJECT_ROOT
        )
        
        print(f"   📂 PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"   📂 DATA_ROOT: {DATA_ROOT}")
        print(f"   📂 SCRAPED_DATA_DIR: {SCRAPED_DATA_DIR}")
        print(f"   📂 PARQUET_DATA_DIR: {PARQUET_DATA_DIR}")
        print(f"   ☁️ USE_GOOGLE_DRIVE: {USE_GOOGLE_DRIVE}")
        print(f"   🔑 GOOGLE_DRIVE_FOLDER_ID: {'SET' if GOOGLE_DRIVE_FOLDER_ID else 'NOT SET'}")
        
    except Exception as e:
        print(f"   ❌ Config import failed: {e}")
        return False
    
    # 2. Check what Railway will do
    print("\n🚂 RAILWAY PLATFORM BEHAVIOR:")
    print("   ✅ Railway provides ephemeral filesystem")
    print("   ✅ Files written during runtime exist temporarily")
    print("   ❌ Files are lost when container restarts/redeploys")
    print("   ⚠️ Local storage is NOT persistent on Railway")
    
    # 3. Analyze storage options
    print("\n💾 STORAGE ANALYSIS:")
    
    if USE_GOOGLE_DRIVE and GOOGLE_DRIVE_FOLDER_ID:
        print("   ✅ GOOGLE DRIVE CONFIGURED:")
        print("     • Data will be saved to Google Drive")
        print("     • Data persists across Railway restarts")
        print("     • This is the RECOMMENDED setup for Railway")
        storage_persistent = True
    else:
        print("   ⚠️ GOOGLE DRIVE NOT CONFIGURED:")
        print("     • Data will only be saved locally on Railway")
        print("     • Data will be LOST when Railway container restarts")
        print("     • This is NOT recommended for production")
        storage_persistent = False
    
    # 4. Check scraper functionality
    print("\n🔄 SCRAPER FUNCTIONALITY ON RAILWAY:")
    
    try:
        from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY
        binance_configured = bool(BINANCE_API_KEY and BINANCE_SECRET_KEY)
        print(f"   🔑 BINANCE API KEYS: {'CONFIGURED' if binance_configured else 'MISSING'}")
        
        if binance_configured:
            print("   ✅ Scraper will work on Railway")
            print("   ✅ Can collect real-time crypto data")
            print("   ✅ Binance API calls will function normally")
        else:
            print("   ❌ Scraper needs BINANCE_API_KEY and BINANCE_SECRET_KEY")
            print("   ❌ Set these as Railway environment variables")
            
    except Exception as e:
        print(f"   ⚠️ Could not check Binance config: {e}")
        binance_configured = False
    
    # 5. Check current production server
    print("\n🖥️ PRODUCTION SERVER ANALYSIS:")
    
    if Path("production_server.py").exists():
        print("   ✅ production_server.py exists")
        print("   ✅ Railway will run this via Procfile")
        print("   ✅ Health endpoint will be available")
        
        # Check if scraper is part of production server
        with open("production_server.py", "r") as f:
            content = f.read()
            if "scraper" in content.lower() or "data_collector" in content.lower():
                print("   ✅ Scraper integration detected in production server")
                runs_scraper = True
            else:
                print("   ⚠️ Production server doesn't appear to run scraper")
                print("   ⚠️ You may need to manually start data collection")
                runs_scraper = False
    else:
        print("   ❌ production_server.py not found")
        runs_scraper = False
    
    # 6. Overall assessment
    print("\n🎯 DEPLOYMENT ASSESSMENT:")
    
    issues = []
    if not binance_configured:
        issues.append("Missing Binance API keys")
    if not storage_persistent:
        issues.append("No persistent storage configured")
    if not runs_scraper:
        issues.append("Scraper not integrated in production server")
    
    if not issues:
        print("   🎉 READY FOR DEPLOYMENT!")
        print("   ✅ All systems configured correctly")
        print("   ✅ Data will persist via Google Drive")
        print("   ✅ Scraper will collect data automatically")
    else:
        print("   ⚠️ DEPLOYMENT ISSUES DETECTED:")
        for issue in issues:
            print(f"     • {issue}")
    
    # 7. Recommendations
    print("\n📋 RAILWAY DEPLOYMENT RECOMMENDATIONS:")
    
    print("   1. SET ENVIRONMENT VARIABLES IN RAILWAY:")
    print("      • BINANCE_API_KEY=your_api_key")
    print("      • BINANCE_SECRET_KEY=your_secret_key")
    if not USE_GOOGLE_DRIVE:
        print("      • USE_GOOGLE_DRIVE=true")
        print("      • GOOGLE_DRIVE_FOLDER_ID=your_folder_id")
    
    print("\n   2. GOOGLE DRIVE SETUP (if not done):")
    print("      • Create a Google Drive folder for data storage")
    print("      • Upload credentials.json to Railway (as secret file)")
    print("      • Set GOOGLE_DRIVE_FOLDER_ID environment variable")
    
    print("\n   3. VERIFY SCRAPER INTEGRATION:")
    if not runs_scraper:
        print("      • Add scraper startup to production_server.py, OR")
        print("      • Create separate Railway service for data collection")
    
    print("\n   4. MONITORING:")
    print("      • Check Railway logs after deployment")
    print("      • Monitor /health endpoint")
    print("      • Verify data appears in Google Drive")
    
    return len(issues) == 0

def main():
    """Run the analysis"""
    success = analyze_railway_deployment()
    
    print("\n" + "=" * 60)
    if success:
        print("🚀 READY TO DEPLOY TO RAILWAY!")
        print("Your scraper will work and data will be persistent.")
    else:
        print("⚠️ FIX ISSUES BEFORE DEPLOYING TO RAILWAY")
        print("Data may be lost without proper configuration.")
    print("=" * 60)

if __name__ == "__main__":
    main()

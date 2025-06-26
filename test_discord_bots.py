#!/usr/bin/env python3
"""
Discord Bot Test Script
Test each Discord bot individually to identify issues
"""
import sys
import os
import traceback
from pathlib import Path

# Add the root directory to Python path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

def test_trading_bot():
    """Test Trading Discord Bot"""
    print("🔄 Testing Trading Discord Bot...")
    try:
        from src.trading_bot.discord_trader_bot import bot as trading_bot
        print("✅ Trading bot imported successfully")
        return True
    except Exception as e:
        print(f"❌ Trading bot import failed: {e}")
        traceback.print_exc()
        return False

def test_data_collector_bot():
    """Test Data Collector Discord Bot"""
    print("🔄 Testing Data Collector Discord Bot...")
    try:
        from src.data_collector.discord_bot import bot as data_bot
        print("✅ Data collector bot imported successfully")
        return True
    except Exception as e:
        print(f"❌ Data collector bot import failed: {e}")
        traceback.print_exc()
        return False

def test_training_bot():
    """Test Training Discord Bot"""
    print("🔄 Testing Training Discord Bot...")
    try:
        from src.model_training.discord_training_bot import bot as training_bot
        print("✅ Training bot imported successfully")
        return True
    except Exception as e:
        print(f"❌ Training bot import failed: {e}")
        traceback.print_exc()
        return False

def test_data_scraper():
    """Test data scraper import"""
    print("🔄 Testing Data Scraper module...")
    try:
        from src.data_collector.data_scraper import main as start_scraper
        print("✅ Data scraper imported successfully")
        return True
    except Exception as e:
        print(f"❌ Data scraper import failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("🤖 DISCORD BOT IMPORT TESTING")
    print("="*60)
    
    results = []
    results.append(("Trading Bot", test_trading_bot()))
    results.append(("Data Scraper", test_data_scraper()))
    results.append(("Data Collector Bot", test_data_collector_bot()))
    results.append(("Training Bot", test_training_bot()))
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

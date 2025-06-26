#!/usr/bin/env python3
"""
Test script to verify all modules can be imported in Docker environment
This simulates the import behavior of the Discord bot
"""

import os
import sys
import logging

# Add project root to path for imports (simulate Docker environment)
sys.path.append('/app')
sys.path.append('/app/src')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModuleTest")

def test_trading_imports():
    """Test trading module imports"""
    try:
        from src.trading_bot.trade_runner import run_single_trade, get_usdt_balance
        logger.info("✅ Trading modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Trading modules failed: {e}")
        return False

def test_scraper_imports():
    """Test scraper module imports"""
    try:
        from src.data_collector.data_scraper import main as start_scraper
        logger.info("✅ Scraper modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Scraper modules failed: {e}")
        return False

def test_model_training_imports():
    """Test model training imports"""
    try:
        from src.model_training.random_forest_trainer import main as train_rf_model
        logger.info("✅ Model training modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Model training modules failed: {e}")
        return False

def test_stats_imports():
    """Test stats module imports"""
    try:
        from trading_stats import get_stats_manager
        logger.info("✅ Trading stats modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Trading stats modules failed: {e}")
        return False

def test_discord_imports():
    """Test Discord bot imports"""
    try:
        import discord
        from discord.ext import commands
        logger.info("✅ Discord modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Discord modules failed: {e}")
        return False

def main():
    """Run all import tests"""
    logger.info("🔍 Testing module imports...")
    
    results = {
        "discord": test_discord_imports(),
        "trading": test_trading_imports(),
        "scraper": test_scraper_imports(),
        "model_training": test_model_training_imports(),
        "stats": test_stats_imports()
    }
    
    logger.info("\n" + "="*50)
    logger.info("📊 IMPORT TEST RESULTS:")
    for module, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {module}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    logger.info(f"\n🎯 Overall: {total_passed}/{total_tests} modules imported successfully")
    
    if total_passed == total_tests:
        logger.info("🎉 All modules are working - bot should be fully functional!")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - total_passed} modules failed - bot will have limited functionality")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

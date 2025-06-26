#!/usr/bin/env python3
"""
Simple test to check module imports in Docker
"""

import os
import sys

# Add project root to path for imports
sys.path.append('/app')
sys.path.append('/app/src')

print("🔍 Testing module imports in Docker container...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Test Discord
try:
    import discord
    print("✅ Discord imported successfully")
except Exception as e:
    print(f"❌ Discord failed: {e}")

# Test trading modules
try:
    from src.trading_bot.trade_runner import run_single_trade, get_usdt_balance
    print("✅ Trading modules imported successfully")
except Exception as e:
    print(f"❌ Trading modules failed: {e}")

# Test scraper modules
try:
    from src.data_collector.data_scraper import main as start_scraper
    print("✅ Scraper modules imported successfully")
except Exception as e:
    print(f"❌ Scraper modules failed: {e}")

# Test model training
try:
    from src.model_training.random_forest_trainer import main as train_rf_model
    print("✅ Model training modules imported successfully")
except Exception as e:
    print(f"❌ Model training modules failed: {e}")

# Test trading stats
try:
    from src.trading_stats import get_stats_manager
    print("✅ Trading stats modules imported successfully")
except Exception as e:
    print(f"❌ Trading stats modules failed: {e}")

print("🎯 Module import test completed!")

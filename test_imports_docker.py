#!/usr/bin/env python3
"""
Test script to verify module imports work in Docker environment
"""
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append('/app')  # Docker working directory

print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

if os.path.exists('src'):
    print(f"Files in src/: {os.listdir('src')}")
else:
    print("src/ directory not found!")

# Test trading bot import
try:
    from src.trading_bot.trade_runner import TradingBot
    print("✅ Trading bot import successful")
except ImportError as e:
    print(f"❌ Trading bot import failed: {e}")

# Test data collector import  
try:
    from src.data_collector.data_scraper import main as start_scraper
    print("✅ Data scraper import successful")
except ImportError as e:
    print(f"❌ Data scraper import failed: {e}")

# Test model training import
try:
    from src.model_training.random_forest_trainer import main as train_rf_model
    print("✅ Model training import successful")
except ImportError as e:
    print(f"❌ Model training import failed: {e}")

print("\nImport test completed!")

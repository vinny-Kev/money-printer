#!/usr/bin/env python3
"""
Local Trading System Setup - No Docker Required!
Simple setup to get your trading bot working locally
"""
import os
import sys
from pathlib import Path

print("🚀 **MONEY PRINTER LOCAL SETUP**")
print("=" * 50)

# Check if we're in the right directory
if not Path("main.py").exists():
    print("❌ Please run this from the money-printer directory")
    sys.exit(1)

print("📍 Running from correct directory")

# Install required packages
print("📦 Installing required packages...")
required_packages = [
    "python-binance",
    "discord.py", 
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib",
    "python-dotenv",
    "pyarrow",
    "requests"
]

try:
    import subprocess
    for package in required_packages:
        print(f"   Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                      capture_output=True, check=True)
    print("✅ All packages installed!")
except Exception as e:
    print(f"⚠️ Package installation issue: {e}")

# Check data availability
print("\n📊 Checking data availability...")
data_dir = Path("data/scraped_data/parquet_files")
if data_dir.exists():
    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"✅ Found {len(parquet_files)} data files")
else:
    print("⚠️ No data directory found - you'll need to run data collection first")

# Check models
print("\n🤖 Checking models...")
models_dir = Path("data/models")
if models_dir.exists():
    model_files = list(models_dir.glob("**/*.pkl"))
    print(f"✅ Found {len(model_files)} trained models")
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"   📁 {model_file.relative_to(models_dir)} ({size_mb:.1f} MB)")
else:
    print("⚠️ No models directory found")

# Test basic imports
print("\n🔧 Testing core functionality...")
try:
    sys.path.append(str(Path.cwd() / "src"))
    from src.config import DATA_ROOT, MODELS_DIR
    print("✅ Config imports working")
except Exception as e:
    print(f"❌ Config import failed: {e}")

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    print("✅ Core ML libraries working")
except Exception as e:
    print(f"❌ ML libraries failed: {e}")

# Available commands
print("\n🎮 **AVAILABLE COMMANDS:**")
print("=" * 50)
print("📊 Data Collection:")
print("   python main.py collect")
print("   python main.py collect --hours 2")
print("")
print("🤖 Model Training:")
print("   python main.py train")
print("   python check_model_status.py")
print("")
print("💰 Trading:")
print("   python main.py trade")
print("   python main.py status")
print("")
print("🤖 Discord Bot:")
print("   python main.py discord-trade")
print("   python main.py discord-data")
print("")
print("🔍 Health Monitoring:")
print("   python production_health_monitor.py")

print("\n🎯 **QUICK START:**")
print("=" * 50)
print("1. python main.py status          # Check system")
print("2. python check_model_status.py   # Check models")
print("3. python main.py trade           # Start trading")
print("")
print("🚀 **NO DOCKER NEEDED - EVERYTHING RUNS LOCALLY!**")

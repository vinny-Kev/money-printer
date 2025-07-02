#!/usr/bin/env python3
"""
Simple test for ensemble trainer components
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing individual imports...")

# Test basic imports
print("1. Testing basic imports...")
try:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    print("   ✅ Basic imports OK")
except Exception as e:
    print(f"   ❌ Basic imports failed: {e}")

# Test sklearn
print("2. Testing sklearn...")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    print("   ✅ Sklearn imports OK")
except Exception as e:
    print(f"   ❌ Sklearn imports failed: {e}")

# Test XGBoost and LightGBM
print("3. Testing XGBoost...")
try:
    import xgboost as xgb
    print("   ✅ XGBoost OK")
except Exception as e:
    print(f"   ❌ XGBoost failed: {e}")

print("4. Testing LightGBM...")
try:
    import lightgbm as lgb
    print("   ✅ LightGBM OK")
except Exception as e:
    print(f"   ❌ LightGBM failed: {e}")

# Test config
print("5. Testing config...")
try:
    from src.config import MODELS_DIR, RANDOM_STATE
    print("   ✅ Config OK")
except Exception as e:
    print(f"   ❌ Config failed: {e}")

# Test minimal storage
print("6. Testing minimal storage...")
try:
    from src.storage.minimal_storage_manager import MinimalStorageManager
    print("   ✅ Minimal storage OK")
except Exception as e:
    print(f"   ❌ Minimal storage failed: {e}")

print("\nAll tests completed!")

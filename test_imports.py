"""
Quick test to isolate import issues
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

print("Testing imports...")

try:
    from config import TRADING_ENABLED, BINANCE_TESTNET
    print("✅ Config imports working")
    print(f"   TRADING_ENABLED: {TRADING_ENABLED}")
    print(f"   BINANCE_TESTNET: {BINANCE_TESTNET}")
except Exception as e:
    print(f"❌ Config import error: {e}")

try:
    from storage.enhanced_storage_manager import EnhancedStorageManager
    print("✅ Storage manager import working")
except Exception as e:
    print(f"❌ Storage manager import error: {e}")

try:
    from binance_wrapper import EnhancedBinanceClient
    print("✅ Binance wrapper import working")
except Exception as e:
    print(f"❌ Binance wrapper import error: {e}")

print("Import test complete")

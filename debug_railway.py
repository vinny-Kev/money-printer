#!/usr/bin/env python3
"""Debug script to check Railway environment"""
import os
import sys

print("=== RAILWAY ENVIRONMENT DEBUG ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print(f"__file__: {__file__}")
print(f"Directory contents: {os.listdir('.')}")

if os.path.exists('src'):
    print(f"src/ contents: {os.listdir('src')}")
    if os.path.exists('src/trading_bot'):
        print(f"src/trading_bot/ contents: {os.listdir('src/trading_bot')}")
    if os.path.exists('src/data_collector'):
        print(f"src/data_collector/ contents: {os.listdir('src/data_collector')}")
    if os.path.exists('src/model_training'):
        print(f"src/model_training/ contents: {os.listdir('src/model_training')}")

print("=== IMPORT TESTS ===")
try:
    import trading_bot
    print("✅ trading_bot import: SUCCESS")
except ImportError as e:
    print(f"❌ trading_bot import: {e}")

try:
    import src.trading_bot
    print("✅ src.trading_bot import: SUCCESS")
except ImportError as e:
    print(f"❌ src.trading_bot import: {e}")

try:
    from src.trading_bot import trade_runner
    print("✅ src.trading_bot.trade_runner import: SUCCESS")
except ImportError as e:
    print(f"❌ src.trading_bot.trade_runner import: {e}")

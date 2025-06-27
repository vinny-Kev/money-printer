#!/usr/bin/env python3
"""
Quick setup script to enable live trading mode
"""

import os
from dotenv import load_dotenv, set_key

def enable_live_trading():
    """Enable live trading mode in environment."""
    
    env_file = ".env"
    
    # Load current environment
    load_dotenv()
    
    print("🔄 Enabling Live Trading Mode...")
    
    # Set LIVE_TRADING to true
    set_key(env_file, "LIVE_TRADING", "true")
    
    print("✅ LIVE_TRADING=true set in .env file")
    print("🚀 Live trading mode enabled!")
    print("")
    print("📋 Next steps:")
    print("1. Restart the Discord bot")
    print("2. Use /balance command - should show real USDT balance")
    print("3. Make sure your Binance deposit is confirmed first")
    print("")
    print("⚠️ WARNING: This enables REAL MONEY trading!")
    print("   Only enable after confirming your deposit is successful")

if __name__ == "__main__":
    enable_live_trading()

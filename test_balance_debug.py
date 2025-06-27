#!/usr/bin/env python3
"""
Quick test to debug balance fetching issue
"""
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Testing balance fetching...")

try:
    from trading_bot.trade_runner import get_account_balance_safe
    print("✅ Successfully imported get_account_balance_safe")
    
    # Call the function
    result = get_account_balance_safe()
    print(f"✅ Function call successful")
    print(f"📊 Result: {result}")
    
    # Check result details
    print(f"\n🔍 Analysis:")
    print(f"  Status: {result.get('status', 'unknown')}")
    print(f"  Balance: ${result.get('balance', 0):.2f}")
    print(f"  Mode: {result.get('mode', 'unknown')}")
    print(f"  Message: {result.get('message', 'no message')}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

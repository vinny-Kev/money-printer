#!/usr/bin/env python3
"""
Debug Emergency Stop Test
"""
import sys
import os
sys.path.append('src')

from src.safe_config import get_config
from src.trading_safety import TradingSafetyManager

# Test emergency stop mechanism
stop_flag_path = "TRADING_DISABLED.flag"

# Create emergency stop flag
with open(stop_flag_path, "w") as f:
    f.write("Test emergency stop")

print(f"Created flag file: {stop_flag_path}")
print(f"Flag file exists: {os.path.exists(stop_flag_path)}")

# Check if safety manager respects the flag
config = get_config()
safety_mgr = TradingSafetyManager(config)
can_trade, reason = safety_mgr.can_trade_now()

print(f"Can trade: {can_trade}")
print(f"Reason: '{reason}'")
print(f"Reason lower: '{reason.lower()}'")
print(f"Contains 'disabled': {'disabled' in reason.lower()}")
print(f"Contains 'flag file': {'flag file' in reason.lower()}")
print(f"Test condition: {not can_trade and ('disabled' in reason.lower() or 'flag file' in reason.lower())}")

# Test config validation directly
print(f"\nTesting config.validate_runtime_safety():")
runtime_safe = config.validate_runtime_safety()
print(f"Runtime safe: {runtime_safe}")

# Cleanup
if os.path.exists(stop_flag_path):
    os.remove(stop_flag_path)
    print(f"Cleaned up flag file")

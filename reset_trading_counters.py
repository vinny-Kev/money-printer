#!/usr/bin/env python3
"""
Reset Daily Trade Counter and Enable Full Trading
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def reset_daily_trades():
    """Reset daily trade counters"""
    print("🔄 Resetting Daily Trade Counters")
    print("=" * 40)
    
    try:
        from safe_config import get_config
        from trading_safety import TradingSafetyManager
        
        config = get_config()
        safety_mgr = TradingSafetyManager(config)
        
        print(f"📊 Current daily trades: {safety_mgr.daily_trade_count}/{config.max_daily_trades}")
        
        # Reset counters
        safety_mgr.daily_trade_count = 0
        safety_mgr.hourly_trade_count = 0
        safety_mgr.total_bot_pnl = 0.0
        safety_mgr.daily_pnl = 0.0
        
        # Reset all symbol states
        for symbol in safety_mgr.trade_states:
            state = safety_mgr.trade_states[symbol]
            state.daily_trade_count = 0
            state.hourly_trade_count = 0
            state.consecutive_losses = 0
            state.locked_until = None
            state.is_active = False
        
        # Save the reset state
        safety_mgr.save_state()
        
        print(f"✅ Reset complete!")
        print(f"📈 Daily trades: {safety_mgr.daily_trade_count}/{config.max_daily_trades}")
        print(f"⏰ Hourly trades: {safety_mgr.hourly_trade_count}/{config.max_hourly_trades}")
        print(f"💰 PnL reset: ${safety_mgr.total_bot_pnl}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    reset_daily_trades()

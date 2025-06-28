#!/usr/bin/env python3
"""
Manual Reset of Trading State File
"""

import os
import json
from datetime import datetime

def reset_trading_state():
    """Manually reset the trading state file"""
    print("🔄 Manually Resetting Trading State")
    print("=" * 40)
    
    state_file = "data/trading_state.json"
    
    # Check if state file exists
    if os.path.exists(state_file):
        print(f"📁 Found existing state file: {state_file}")
        
        # Read current state
        try:
            with open(state_file, 'r') as f:
                current_state = json.load(f)
            
            print(f"📊 Current daily trades: {current_state.get('daily_trade_count', 0)}")
            print(f"📊 Current hourly trades: {current_state.get('hourly_trade_count', 0)}")
            print(f"💰 Current PnL: ${current_state.get('total_bot_pnl', 0)}")
        except Exception as e:
            print(f"⚠️ Could not read state file: {e}")
            current_state = {}
    else:
        print(f"📁 No existing state file found")
        current_state = {}
    
    # Create new reset state
    reset_state = {
        "daily_trade_count": 0,
        "hourly_trade_count": 0,
        "total_bot_pnl": 0.0,
        "daily_pnl": 0.0,
        "daily_winning_trades": 0,
        "daily_losing_trades": 0,
        "starting_balance": current_state.get('starting_balance', 100.0),
        "last_save": datetime.utcnow().isoformat(),
        "trade_states": {}
    }
    
    # Reset all symbol states if they exist
    if 'trade_states' in current_state:
        for symbol, state in current_state['trade_states'].items():
            reset_state['trade_states'][symbol] = {
                'is_active': False,
                'last_trade_time': None,
                'consecutive_losses': 0,
                'daily_trade_count': 0,
                'total_pnl': 0.0,
                'locked_until': None
            }
            print(f"   🔄 Reset {symbol} state")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Write reset state
    try:
        with open(state_file, 'w') as f:
            json.dump(reset_state, f, indent=2)
        
        print(f"✅ Reset complete!")
        print(f"📈 Daily trades: 0")
        print(f"⏰ Hourly trades: 0") 
        print(f"💰 PnL: $0.00")
        print(f"🎯 Trading system ready!")
        
        return True
    except Exception as e:
        print(f"❌ Error writing reset state: {e}")
        return False

if __name__ == "__main__":
    reset_trading_state()

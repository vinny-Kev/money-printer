#!/usr/bin/env python3
import sys
sys.path.append('src')
from src.safe_config import get_config
from src.trading_safety import TradingSafetyManager
from datetime import datetime

config = get_config()
safety_mgr = TradingSafetyManager(config)

print(f'Max daily trades: {config.max_daily_trades}')
print(f'Current daily count: {safety_mgr.daily_trade_count}')

# Set to limit
safety_mgr.daily_trade_count = config.max_daily_trades
safety_mgr.last_day_reset = datetime.utcnow()
safety_mgr._save_state()

can_trade, reason = safety_mgr.can_trade_now()
print(f'Can trade: {can_trade}')
print(f'Reason: {reason}')
print(f'Reason lower: {reason.lower()}')
print(f'Contains "trade limit": {"trade limit" in reason.lower()}')
print(f'Test condition: {not can_trade and "trade limit" in reason.lower()}')

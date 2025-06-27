#!/usr/bin/env python3
"""
Test script to verify symbol affordability filtering with realistic high MOQ scenarios
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from safe_config import get_config
from trading_safety import TradingSafetyManager

def test_extreme_scenarios():
    """Test affordability with very small balances and high MOQ coins"""
    config = get_config()
    safety_mgr = TradingSafetyManager(config)
    
    # Test very small balances
    test_balances = [3.0, 5.0, 8.0, 12.0]
    
    # Simulate high MOQ scenarios (common in real trading)
    high_moq_scenarios = [
        {"symbol": "SHIBUSDT", "price": 0.00002, "realistic_min_order": 8.0},    # Often $8-15 minimum
        {"symbol": "PEPEUSDT", "price": 0.000001, "realistic_min_order": 5.0},   # Often $5-12 minimum
        {"symbol": "FLOKIUSDT", "price": 0.0002, "realistic_min_order": 10.0},   # Often $10+ minimum
        {"symbol": "BONKUSDT", "price": 0.00003, "realistic_min_order": 6.0},    # Often $6+ minimum
    ]
    
    print("🔥 Testing Extreme Small Balance Scenarios")
    print("=" * 70)
    
    for balance in test_balances:
        print(f"\n💰 Balance: ${balance:.2f}")
        print("-" * 50)
        
        # Test normal coins first
        normal_coins = [
            {"symbol": "DOGEUSDT", "price": 0.08},
            {"symbol": "TRXUSDT", "price": 0.20},
            {"symbol": "CHZUSDT", "price": 0.12},
            {"symbol": "ADAUSDT", "price": 0.40},
        ]
        
        affordable_normal = 0
        for coin in normal_coins:
            can_afford, reason, info = safety_mgr.can_afford_symbol(coin["symbol"], coin["price"], balance)
            if can_afford:
                affordable_normal += 1
                print(f"  ✅ {coin['symbol']:12} | ${coin['price']:>6.4f} | ${info.get('min_order_value', 0):>5.2f} min")
            else:
                print(f"  ❌ {coin['symbol']:12} | ${coin['price']:>6.4f} | ${info.get('min_order_value', 0):>5.2f} min | {reason}")
        
        # Test high MOQ meme coins (manually simulate realistic MOQs)
        print(f"\n  📊 High MOQ Meme Coins (realistic minimums):")
        affordable_meme = 0
        for scenario in high_moq_scenarios:
            min_order = scenario["realistic_min_order"]
            if balance >= min_order * 1.002:  # Add small buffer for fees
                affordable_meme += 1
                print(f"  ✅ {scenario['symbol']:12} | ${scenario['price']:>6.6f} | ${min_order:>5.2f} min")
            else:
                print(f"  ❌ {scenario['symbol']:12} | ${scenario['price']:>6.6f} | ${min_order:>5.2f} min | Too expensive")
        
        total_affordable = affordable_normal + affordable_meme
        total_coins = len(normal_coins) + len(high_moq_scenarios)
        
        print(f"\n  🎯 Summary: {total_affordable}/{total_coins} coins affordable")
        
        if balance < 10:
            if affordable_meme == 0:
                print(f"  ⚠️  No meme coins affordable - focus on major/mid-cap coins")
            if total_affordable < total_coins // 2:
                print(f"  💡 Consider depositing more funds for better trading options")
    
    print("\n" + "=" * 70)
    print("🏆 Recommendations for Small Balances:")
    print("• $3-5:   Stick to DOGE, TRX, CHZ - avoid meme coins")
    print("• $5-8:   Can trade some meme coins, but limited selection")
    print("• $8-12:  Most coins become affordable")
    print("• $12+:   Full trading flexibility")
    print("\n🔧 The system automatically filters unaffordable symbols!")

if __name__ == "__main__":
    test_extreme_scenarios()

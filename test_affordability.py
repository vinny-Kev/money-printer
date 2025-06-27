#!/usr/bin/env python3
"""
Test script to verify symbol affordability filtering for small balances
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from safe_config import get_config
from trading_safety import TradingSafetyManager

def test_affordability():
    """Test affordability checking for different balance sizes"""
    config = get_config()
    safety_mgr = TradingSafetyManager(config)
    
    # Test scenarios
    test_cases = [
        {"balance": 5.0, "name": "Very Small Balance ($5)"},
        {"balance": 10.0, "name": "Small Balance ($10)"},
        {"balance": 25.0, "name": "Medium Small Balance ($25)"},
        {"balance": 100.0, "name": "Larger Balance ($100)"},
    ]
    
    # Sample symbols with different characteristics (more realistic MOQs)
    test_symbols = [
        {"symbol": "BTCUSDT", "price": 45000.0},  # High price, low MOQ
        {"symbol": "ETHUSDT", "price": 3000.0},   # High price, low MOQ
        {"symbol": "ADAUSDT", "price": 0.40},     # Low price, medium MOQ
        {"symbol": "DOGEUSDT", "price": 0.08},    # Low price, low MOQ
        {"symbol": "SHIBUSDT", "price": 0.00002}, # Very low price, HIGH MOQ (realistic: $5-10 min)
        {"symbol": "PEPEUSDT", "price": 0.000001}, # Very low price, VERY HIGH MOQ (realistic: $3-8 min)
        {"symbol": "CHZUSDT", "price": 0.12},     # Low price, reasonable MOQ
        {"symbol": "SOLUSDT", "price": 180.0},    # Medium price, low MOQ
        {"symbol": "XRPUSDT", "price": 0.60},     # Medium price, higher MOQ due to popularity
        {"symbol": "TRXUSDT", "price": 0.20},     # Low price, low MOQ (good for small balances)
    ]
    
    print("🧪 Testing Symbol Affordability for Different Balance Sizes")
    print("=" * 80)
    
    for test_case in test_cases:
        balance = test_case["balance"]
        name = test_case["name"]
        
        print(f"\n💰 {name}")
        print("-" * 50)
        
        affordable_count = 0
        total_count = 0
        
        for symbol_data in test_symbols:
            symbol = symbol_data["symbol"]
            price = symbol_data["price"]
            
            can_afford, reason, info = safety_mgr.can_afford_symbol(symbol, price, balance)
            total_count += 1
            
            if can_afford:
                affordable_count += 1
                status = "✅"
                color = "\033[92m"  # Green
            else:
                status = "❌"
                color = "\033[91m"  # Red
            
            reset_color = "\033[0m"
            
            min_order_value = info.get('min_order_value', 0)
            balance_percent = info.get('balance_percent', 0)
            
            print(f"  {status} {color}{symbol:12}{reset_color} | ${price:>8.4f} | Min: ${min_order_value:>6.2f} | Uses: {balance_percent:>5.1f}% | {reason}")
        
        print(f"\n📊 Result: {affordable_count}/{total_count} symbols affordable ({affordable_count/total_count*100:.1f}%)")
        
        if balance <= 20 and affordable_count < total_count // 2:
            print(f"⚠️  Warning: Small balance limits trading options significantly")
            print(f"💡 Consider: Focus on coins like DOGE, CHZ, ADA for small balances")
    
    print("\n" + "=" * 80)
    print("🎯 Key Takeaways:")
    print("• Very small balances ($5-10) should avoid meme coins (SHIB, PEPE)")
    print("• Medium-cap coins (SOL, ADA, DOGE) are better for small balances")
    print("• Major coins (BTC, ETH) work well due to low minimum quantities")
    print("• The system will automatically filter out unaffordable symbols")

if __name__ == "__main__":
    test_affordability()

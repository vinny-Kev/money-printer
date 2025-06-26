#!/usr/bin/env python3
"""
Test Trading Script - Money Printer

This script allows you to easily test the trading bot with different amounts
and see how it performs with various budget sizes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trading_bot.trade_runner import main, DRY_TRADE_BUDGET

def test_with_amounts():
    """
    Interactive testing with different trading amounts
    """
    print("🔥" + "="*50 + "🔥")
    print("   MONEY PRINTER - TRADING TEST SUITE")
    print("🔥" + "="*50 + "🔥")
    print()
    
    test_amounts = [
        ("Small Budget", 50),
        ("Medium Budget", 500), 
        ("Large Budget", 5000),
        ("Whale Budget", 50000),
        ("Custom Amount", None)
    ]
    
    print("Select a test amount:")
    for i, (name, amount) in enumerate(test_amounts, 1):
        if amount:
            print(f"  {i}. {name}: ${amount:,}")
        else:
            print(f"  {i}. {name}: Enter your own amount")
    
    print(f"  0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                return
            
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(test_amounts):
                print("❌ Invalid choice. Please try again.")
                continue
                
            name, amount = test_amounts[choice_idx]
            
            if amount is None:  # Custom amount
                while True:
                    try:
                        custom_amount = float(input("Enter custom trading amount: $").strip())
                        if custom_amount < 3:
                            print("❌ Amount must be at least $3")
                            continue
                        amount = custom_amount
                        name = f"Custom (${amount:,.2f})"
                        break
                    except ValueError:
                        print("❌ Please enter a valid number")
                        continue
            
            print(f"\n🚀 Testing with {name}...")
            print("=" * 60)
            
            # Set the global budget for this test
            import src.trading_bot.trade_runner as tr
            tr.dry_trade_budget = amount
            tr.DRY_TRADE_BUDGET = amount
            
            # Run the trading bot
            main()
            
            print("=" * 60)
            print("✅ Test completed!")
            
            # Ask if user wants to run another test
            another = input("\nRun another test? (y/n): ").strip().lower()
            if another not in ['y', 'yes']:
                break
                
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\n👋 Testing cancelled by user.")
            return
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            continue

def show_data_source_info():
    """
    Display information about data sources
    """
    print("\n📊 DATA SOURCE INFORMATION:")
    print("=" * 50)
    print("🔴 LIVE MODE:")
    print("  • Uses real Binance API")
    print("  • Real market data via WebSocket")
    print("  • Actual trades with real money")
    print("  • All fees and slippage apply")
    print()
    print("🟡 DRY/TEST MODE:")
    print("  • Uses Binance Testnet API")
    print("  • Live market data but simulated trades")
    print("  • No real money involved")
    print("  • Perfect for testing strategies")
    print()
    print("📈 MARKET DATA:")
    print("  • Fetches top 200 USDT pairs by volume")
    print("  • Uses 1-hour candlestick data")
    print("  • 50 periods for technical analysis")
    print("  • Real-time price data")
    print("=" * 50)

if __name__ == "__main__":
    try:
        show_data_source_info()
        test_with_amounts()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

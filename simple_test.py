#!/usr/bin/env python3
"""
Simple Direct Test - Money Printer Trading Bot

This script provides a simple way to test the trading bot with different amounts.
No complex menus - just direct testing.
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_trading_with_amount(amount):
    """
    Test trading with a specific amount
    """
    print(f"\n🔥 TESTING WITH ${amount:,.2f} 🔥")
    print("=" * 60)
    
    # Import here to avoid issues with path
    try:
        from trading_bot.trade_runner import main
        import trading_bot.trade_runner as tr
        
        # Set the budget
        tr.dry_trade_budget = amount
        tr.DRY_TRADE_BUDGET = amount
        
        # Override the user input function to auto-return the amount
        def mock_get_user_trading_budget():
            print(f"✅ Auto-setting budget to: ${amount:.2f}")
            return amount
        
        # Replace the function temporarily
        original_func = tr.get_user_trading_budget
        tr.get_user_trading_budget = mock_get_user_trading_budget
        
        # Run the test
        main()
        
        # Restore original function
        tr.get_user_trading_budget = original_func
        
        print("=" * 60)
        print(f"✅ Test completed with ${amount:,.2f}")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function with simple test options
    """
    print("🔥" + "="*58 + "🔥")
    print("   MONEY PRINTER - SIMPLE TRADING TEST")
    print("🔥" + "="*58 + "🔥")
    print()
    print("📊 DATA SOURCE INFO:")
    print("  • Uses LIVE market data from Binance")
    print("  • Trades are SIMULATED (Testnet)")
    print("  • No real money involved")
    print("  • Perfect for testing strategies")
    print()
    
    # Simple test amounts
    test_amounts = [50, 500, 5000, 50000]
    
    print("Quick test options:")
    for i, amount in enumerate(test_amounts, 1):
        print(f"  {i}. Test with ${amount:,}")
    print("  5. Custom amount")
    print("  0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice in ["1", "2", "3", "4"]:
                amount = test_amounts[int(choice) - 1]
                test_trading_with_amount(amount)
                
                # Ask if user wants another test
                again = input("\nRun another test? (y/n): ").strip().lower()
                if again not in ['y', 'yes']:
                    break
                    
            elif choice == "5":
                try:
                    custom_amount = float(input("Enter custom amount: $").strip())
                    if custom_amount < 3:
                        print("❌ Amount must be at least $3")
                        continue
                    test_trading_with_amount(custom_amount)
                    
                    # Ask if user wants another test
                    again = input("\nRun another test? (y/n): ").strip().lower()
                    if again not in ['y', 'yes']:
                        break
                except ValueError:
                    print("❌ Please enter a valid number")
                    continue
            else:
                print("❌ Invalid choice. Please try again.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Testing cancelled by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()

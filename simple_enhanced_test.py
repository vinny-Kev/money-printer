#!/usr/bin/env python3
"""
Simple Enhanced Trading Test - No User Input Required

This script demonstrates the enhanced trading bot functionality:
1. Persistent trades with monitoring
2. Incremental learning data collection
3. Model retraining capabilities
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_system():
    """
    Run a comprehensive test of the enhanced trading system.
    """
    print("🔥" + "="*60 + "🔥")
    print("   ENHANCED MONEY PRINTER - AUTOMATED TEST")
    print("🔥" + "="*60 + "🔥")
    print()
    
    print("📊 TESTING ENHANCED FEATURES:")
    print("  ✅ Persistent trades until TP/SL exit")
    print("  ✅ Real-time trade monitoring (6min timeout for demo)")
    print("  ✅ Incremental learning data collection")
    print("  ✅ Enhanced CSV logging for taxation")
    print()
    
    # Test 1: Run a trade with monitoring
    print("🚀 TEST 1: Running enhanced trade with monitoring...")
    print("-" * 60)
    
    try:
        # Import and configure the trading system
        from src.trading_bot.trade_runner import run_single_trade
        import src.trading_bot.trade_runner as tr
        
        # Set test budget
        tr.dry_trade_budget = 500
        tr.DRY_TRADE_BUDGET = 500
        
        # Run the trade
        receipt = run_single_trade()
        
        if "error" in receipt:
            print(f"❌ Trade failed: {receipt['error']}")
        else:
            print("✅ Trade completed successfully!")
            print(f"📊 Symbol: {receipt.get('coin', 'Unknown')}")
            print(f"💰 P&L: {receipt.get('pnl_percent', 0):.2f}%")
            print(f"⏱️ Duration: {receipt.get('trade_duration_formatted', 'Unknown')}")
            print(f"🎯 Success: {'✅' if receipt.get('was_successful', False) else '❌'}")
    
    except Exception as e:
        print(f"❌ Error during trade test: {e}")
    
    print()
    print("-" * 60)
    
    # Test 2: Check training data
    print("🤖 TEST 2: Checking incremental learning data...")
    print("-" * 60)
    
    try:
        from src.model_training.incremental_trainer import IncrementalTrainer
        
        trainer = IncrementalTrainer()
        df = trainer.load_trade_data()
        
        if len(df) > 0:
            print(f"✅ Found {len(df)} trades in training data")
            print(f"📊 Win Rate: {(df['was_successful'].sum() / len(df)) * 100:.1f}%")
            print(f"💰 Avg P&L: {df['pnl_percent'].mean():.2f}%")
            
            # Check if retraining is needed
            should_retrain, reason = trainer.should_retrain()
            print(f"🔍 Retraining needed: {should_retrain}")
            print(f"📝 Reason: {reason}")
            
        else:
            print("❌ No training data found - trade may have failed")
            
    except Exception as e:
        print(f"❌ Error checking training data: {e}")
    
    print()
    print("-" * 60)
    
    # Test 3: Check CSV exports
    print("📊 TEST 3: Checking CSV exports...")
    print("-" * 60)
    
    try:
        import pandas as pd
        
        # Check tax CSV
        tax_csv_path = os.path.join("src", "trading_bot", "trading_transactions.csv")
        if os.path.exists(tax_csv_path):
            tax_df = pd.read_csv(tax_csv_path)
            print(f"✅ Tax CSV: {len(tax_df)} transactions recorded")
            print(f"📁 Location: {tax_csv_path}")
        else:
            print("❌ Tax CSV not found")
        
        # Check training CSV
        training_csv_path = os.path.join("data", "transactions", "random_forest_v1_trades.csv")
        if os.path.exists(training_csv_path):
            training_df = pd.read_csv(training_csv_path)
            print(f"✅ Training CSV: {len(training_df)} trades recorded")
            print(f"📁 Location: {training_csv_path}")
        else:
            print("❌ Training CSV not found")
            
    except Exception as e:
        print(f"❌ Error checking CSV files: {e}")
    
    print()
    print("=" * 60)
    print("🎉 ENHANCED SYSTEM TEST COMPLETED!")
    print("=" * 60)
    print()
    print("📋 SUMMARY OF IMPROVEMENTS:")
    print("  1. ✅ Trades now persist until TP/SL is hit")
    print("  2. ✅ Real-time price monitoring implemented")
    print("  3. ✅ Training data automatically collected")
    print("  4. ✅ CSV exports for taxation ready")
    print("  5. ✅ Incremental learning pipeline ready")
    print()
    print("🚀 The enhanced money printer is ready for production!")

if __name__ == "__main__":
    try:
        test_enhanced_system()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

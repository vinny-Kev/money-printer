#!/usr/bin/env python3
"""
Enhanced Trading Bot Test - With Persistent Trades and Incremental Learning

This script tests the enhanced trading bot that:
1. Monitors trades until TP/SL is hit
2. Logs trade outcomes for incremental learning
3. Can trigger model retraining based on performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trading_bot.trade_runner import main as run_trader, DRY_TRADE_BUDGET
from src.model_training.incremental_trainer import IncrementalTrainer

def test_enhanced_trading():
    """
    Test the enhanced trading system with monitoring and logging.
    """
    print("🔥" + "="*60 + "🔥")
    print("   ENHANCED MONEY PRINTER - PERSISTENT TRADING TEST")
    print("🔥" + "="*60 + "🔥")
    print()
    
    print("📊 ENHANCED FEATURES:")
    print("  ✅ Persistent trades until TP/SL exit")
    print("  ✅ Real-time trade monitoring")
    print("  ✅ Incremental learning data collection")
    print("  ✅ Automatic model retraining capability")
    print("  ✅ Enhanced CSV export for taxation")
    print()
    
    # Test options
    test_options = [
        ("Quick Test Trade", 100),
        ("Medium Test Trade", 500), 
        ("Large Test Trade", 2000),
        ("Incremental Training Demo", None),
        ("Exit", None)
    ]
    
    print("Select test option:")
    for i, (name, amount) in enumerate(test_options, 1):
        if amount:
            print(f"  {i}. {name}: ${amount}")
        else:
            print(f"  {i}. {name}")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            choice_idx = int(choice) - 1
            
            if choice_idx < 0 or choice_idx >= len(test_options):
                print("❌ Invalid choice. Please try again.")
                continue
                
            name, amount = test_options[choice_idx]
            
            if name == "Exit":
                print("👋 Goodbye!")
                return
            elif name == "Incremental Training Demo":
                test_incremental_training()
                return
            else:
                print(f"\n🚀 Running {name}...")
                print("=" * 60)
                
                # Set the trading budget
                import src.trading_bot.trade_runner as tr
                tr.dry_trade_budget = amount
                tr.DRY_TRADE_BUDGET = amount
                
                # Run the enhanced trading bot
                run_trader()
                
                print("=" * 60)
                print("✅ Enhanced test completed!")
                
                # Ask if user wants to check training data
                check_training = input("\nCheck incremental training data? (y/n): ").strip().lower()
                if check_training in ['y', 'yes']:
                    show_training_data_summary()
                
                break
                
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\n👋 Test cancelled by user.")
            return
        except Exception as e:
            print(f"❌ Error during test: {e}")
            continue

def show_training_data_summary():
    """
    Display a summary of collected training data.
    """
    try:
        trainer = IncrementalTrainer()
        df = trainer.load_trade_data()
        
        if len(df) == 0:
            print("\n📊 No training data found yet.")
            return
        
        print(f"\n📊 TRAINING DATA SUMMARY:")
        print("=" * 40)
        print(f"  📈 Total Trades: {len(df)}")
        print(f"  ✅ Successful: {df['was_successful'].sum()}")
        print(f"  ❌ Failed: {(~df['was_successful']).sum()}")
        print(f"  📊 Win Rate: {(df['was_successful'].sum() / len(df)) * 100:.1f}%")
        print(f"  💰 Avg P&L: {df['pnl_percent'].mean():.2f}%")
        print(f"  ⏱️ Avg Duration: {df['trade_duration_secs'].mean()/60:.1f} minutes")
        
        if len(df) >= 5:
            print(f"\n📈 Recent 5 trades:")
            recent = df.tail(5)[['coin', 'was_successful', 'pnl_percent', 'trade_duration_secs']]
            for _, row in recent.iterrows():
                duration_str = f"{row['trade_duration_secs']/60:.1f}m"
                status = "✅" if row['was_successful'] else "❌"
                print(f"    {status} {row['coin']}: {row['pnl_percent']:+.2f}% ({duration_str})")
        
        print("=" * 40)
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")

def test_incremental_training():
    """
    Test the incremental training functionality.
    """
    print("\n🤖 INCREMENTAL TRAINING TEST")
    print("=" * 50)
    
    trainer = IncrementalTrainer()
    
    # Check current status
    print("📊 Checking current training data status...")
    df = trainer.load_trade_data()
    
    if len(df) == 0:
        print("❌ No training data found.")
        print("💡 Run some trades first to generate training data.")
        return
    
    print(f"✅ Found {len(df)} trades in training data")
    
    # Check if retraining is recommended
    should_retrain, reason = trainer.should_retrain()
    print(f"🔍 Retraining needed: {should_retrain}")
    print(f"📝 Reason: {reason}")
    
    if should_retrain:
        confirm = input("\n🚀 Run incremental training? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            print("\n🤖 Starting incremental training...")
            result = trainer.run_incremental_training()
            
            if result["success"]:
                print("✅ Training completed successfully!")
                print(f"📊 Accuracy: {result['accuracy']:.3f}")
                print(f"📊 F1 Score: {result['f1_score']:.3f}")
                print(f"💾 Model saved to: {result['model_path']}")
            else:
                print(f"❌ Training failed: {result['error']}")
        else:
            print("⏭️ Training skipped by user")
    else:
        print("✅ Model performance is good - no retraining needed")

if __name__ == "__main__":
    try:
        test_enhanced_trading()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

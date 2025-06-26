#!/usr/bin/env python3
"""
Demo Incremental Training - Shows how the incremental learning system works

This script demonstrates the incremental training functionality with a lowered
threshold for testing purposes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_training.incremental_trainer import IncrementalTrainer

def demo_incremental_training():
    """
    Demonstrate incremental training with current data.
    """
    print("🤖" + "="*60 + "🤖")
    print("   INCREMENTAL TRAINING SYSTEM DEMONSTRATION")
    print("🤖" + "="*60 + "🤖")
    print()
    
    # Create trainer with lower threshold for demo
    trainer = IncrementalTrainer(model_name="random_forest_v1", min_trades=3)
    
    print("📊 Checking current training data...")
    df = trainer.load_trade_data()
    
    if len(df) == 0:
        print("❌ No training data found.")
        return
    
    print(f"✅ Found {len(df)} trades in training data")
    print(f"📊 Win Rate: {(df['was_successful'].sum() / len(df)) * 100:.1f}%")
    print(f"💰 Avg P&L: {df['pnl_percent'].mean():.2f}%")
    print()
    
    # Show sample data
    print("📋 SAMPLE TRADES:")
    print(df[['coin', 'pnl_percent', 'was_successful', 'confidence']].to_string(index=False))
    print()
    
    # Check retraining eligibility
    should_retrain, reason = trainer.should_retrain(win_rate_threshold=60.0, min_trades_required=3)
    print(f"🔍 Should retrain: {should_retrain}")
    print(f"📝 Reason: {reason}")
    print()
    
    if should_retrain:
        print("🚀 RUNNING INCREMENTAL TRAINING...")
        print("-" * 40)
        
        try:
            result = trainer.run_incremental_training(force=True, win_rate_threshold=60.0)
            
            if result["success"]:
                if result.get("skipped"):
                    print(f"⏭️ Training skipped: {result['reason']}")
                else:
                    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
                    print(f"📊 Model Performance:")
                    print(f"   • Trades Used: {result['trades_used']}")
                    print(f"   • Test Accuracy: {result['accuracy']:.3f}")
                    print(f"   • Test F1 Score: {result['f1_score']:.3f}")
                    print(f"   • Features Used: {len(result['features_used'])}")
                    print(f"💾 Model saved to: {result['model_path']}")
                    print()
                    print("🎯 The model has been updated with real trading data!")
            else:
                print(f"❌ Training failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error during training: {e}")
    else:
        print("ℹ️ No retraining needed at this time")
    
    print()
    print("🎉 INCREMENTAL TRAINING DEMO COMPLETED!")

if __name__ == "__main__":
    try:
        demo_incremental_training()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

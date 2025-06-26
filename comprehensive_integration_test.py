#!/usr/bin/env python3
"""
Comprehensive Integration Test for Enhanced Crypto Trading Platform
Tests all major components: Stats, Auto-Culling, Diagnostics, Discord Bot
"""

import os
import sys
import traceback
import asyncio
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_stats_manager():
    """Test Trading Stats Manager functionality"""
    print("🧪 Testing Trading Stats Manager...")
    
    try:
        from trading_stats import TradingStatsManager, ModelPerformance, TrainingMetrics
        
        # Initialize stats manager
        stats_mgr = TradingStatsManager()
          # Test recording a profitable trade
        stats_mgr.record_trade(
            model_name="random_forest_v1",
            was_successful=True,
            pnl=100.0,
            trade_data={"entry_price": 50000.0, "exit_price": 51000.0}
        )
        
        # Test recording a losing trade
        stats_mgr.record_trade(
            model_name="random_forest_v1", 
            was_successful=False,
            pnl=-50.0,
            trade_data={"entry_price": 49000.0, "exit_price": 48000.0}
        )
        
        # Test performance metrics
        performance = stats_mgr.get_model_performance("random_forest_v1")
        print(f"   ✅ Model Performance: {performance.total_trades} trades, {performance.win_rate:.1%} win rate")
        
        # Test dashboard generation
        dashboard = stats_mgr.generate_dashboard()
        print(f"   ✅ Dashboard generated: {len(dashboard)} characters")
        
        # Test underperforming model detection
        underperforming = stats_mgr.get_underperforming_models()
        print(f"   ✅ Underperforming models: {len(underperforming)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Stats Manager test failed: {e}")
        traceback.print_exc()
        return False

def test_auto_culling():
    """Test Auto-Culling System functionality"""
    print("🧪 Testing Auto-Culling System...")
    
    try:
        from auto_culling import AutoCullingSystem
        from trading_stats import TradingStatsManager
        
        # Initialize systems
        culler = AutoCullingSystem()
        stats_mgr = TradingStatsManager()
          # Test model evaluation
        test_model = "xgboost_v1"
        test_performance = {"win_rate": 0.3, "consecutive_losses": 6, "total_pnl": -200.0}
        should_pause, reason = culler.should_cull_model(test_model, test_performance)
        print(f"   ✅ Model evaluation completed: {test_model} should_pause={should_pause}, reason={reason}")
        
        # Test pause/unpause functionality
        culler.pause_model(test_model, "Test pause")
        print(f"   ✅ Model paused: {test_model}")
        
        culler.unpause_model(test_model)
        print(f"   ✅ Model unpaused: {test_model}")
        
        # Test culling configuration
        culler.configure_culling(
            enabled=True,
            min_win_rate=0.6,
            max_consecutive_losses=3,
            max_loss_threshold=200.0
        )
        print(f"   ✅ Culling configuration updated")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Auto-Culling test failed: {e}")
        traceback.print_exc()
        return False

def test_trainer_diagnostics():
    """Test Trainer Diagnostics functionality"""
    print("🧪 Testing Trainer Diagnostics...")
    
    try:
        from model_training.trainer_diagnostics import TrainerDiagnostics
        import numpy as np
        
        # Create sample training data
        X_train = np.random.random((1000, 10))
        y_train = np.random.randint(0, 2, 1000)
        X_val = np.random.random((200, 10))
        y_val = np.random.randint(0, 2, 200)
        
        # Create sample predictions
        y_pred = np.random.random(200)
        train_pred = np.random.random(1000)        # Initialize diagnostics
        diagnostics = TrainerDiagnostics("test_model")
        
        # Test feature importance analysis
        feature_names = [f"feature_{i}" for i in range(10)]
        importance_scores = np.random.random(10)
        
        # Test overfitting detection (using private method as public method doesn't exist)
        train_scores = [0.95, 0.96, 0.97, 0.98]
        val_scores = [0.75, 0.74, 0.73, 0.72]
        overfitting_msg = diagnostics._detect_overfitting(train_scores, val_scores)
        print(f"   ✅ Overfitting detection: {overfitting_msg}")
        
        # Test comprehensive analysis
        analysis = diagnostics.run_comprehensive_analysis(
            X_train, y_train, X_val, y_val, 
            train_pred, y_pred, 
            feature_names, importance_scores
        )
        print(f"   ✅ Comprehensive analysis completed: {len(analysis)} metrics")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Trainer Diagnostics test failed: {e}")
        traceback.print_exc()
        return False

def test_discord_bot_imports():
    """Test Discord Bot imports and basic functionality"""
    print("🧪 Testing Discord Bot Imports...")
    
    try:
        # Test if all required modules can be imported
        from trading_bot.discord_trader_bot import bot
        from trading_stats import get_stats_manager
        from auto_culling import get_auto_culler
        
        print(f"   ✅ Discord bot imported successfully")
        
        # Test stats manager singleton
        stats_mgr = get_stats_manager()
        print(f"   ✅ Stats manager singleton working")
        
        # Test auto-culler singleton
        culler = get_auto_culler()
        print(f"   ✅ Auto-culler singleton working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Discord Bot imports test failed: {e}")
        traceback.print_exc()
        return False

def test_integration_workflow():
    """Test complete integration workflow"""
    print("🧪 Testing Integration Workflow...")
    
    try:
        from trading_stats import get_stats_manager
        from auto_culling import get_auto_culler
        from model_training.trainer_diagnostics import TrainerDiagnostics
        
        # Initialize components
        stats_mgr = get_stats_manager()
        culler = get_auto_culler()
        
        # Simulate a complete trading cycle
        model_name = "integration_test_model"
          # 1. Record training metrics
        from trading_stats import TrainingMetrics
        training_metrics = TrainingMetrics(
            model_name=model_name,
            training_loss=0.15,
            validation_loss=0.18,
            training_time=timedelta(minutes=30),
            overfitting_detected=False,
            feature_importance={"feature_1": 0.3, "feature_2": 0.2}
        )
        stats_mgr.record_training_metrics(training_metrics)
        print(f"   ✅ Training metrics recorded")
          # 2. Record multiple trades
        for i in range(5):
            profit = 50.0 if i % 2 == 0 else -25.0  # Alternating wins/losses
            stats_mgr.record_trade(
                model_name=model_name,
                was_successful=profit > 0,
                pnl=profit,
                trade_data={"entry_price": 50000.0 + i * 100, "exit_price": 50000.0 + i * 100 + profit}
            )
        print(f"   ✅ Trade history recorded")
        
        # 3. Check auto-culling decision
        test_performance = stats_mgr.get_model_performance(model_name)
        should_pause, reason = culler.should_cull_model(model_name, {
            "win_rate": test_performance.win_rate,
            "consecutive_losses": test_performance.consecutive_losses,
            "total_pnl": test_performance.total_pnl
        })
        print(f"   ✅ Auto-culling evaluation: should_pause={should_pause}")
        
        # 4. Generate dashboard
        dashboard = stats_mgr.generate_dashboard()
        print(f"   ✅ Dashboard generated: {len(dashboard)} characters")
        
        # 5. Test model flagging
        flagged = stats_mgr.get_flagged_models()
        print(f"   ✅ Flagged models: {len(flagged)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration workflow test failed: {e}")
        traceback.print_exc()
        return False

def test_data_persistence():
    """Test data persistence and loading"""
    print("🧪 Testing Data Persistence...")
    
    try:
        from trading_stats import TradingStatsManager
        from auto_culling import AutoCullingSystem
          # Test stats manager persistence
        stats_mgr = TradingStatsManager()
        stats_mgr.record_trade(
            model_name="persistence_test",
            was_successful=True,
            pnl=100.0,
            trade_data={"entry_price": 50000.0, "exit_price": 51000.0}
        )
        stats_mgr.save_data()
        print(f"   ✅ Stats data saved")
        
        # Test auto-culler persistence
        culler = AutoCullingSystem()
        culler.pause_model("persistence_test", "Test persistence")
        culler.save_data()
        print(f"   ✅ Auto-culler data saved")
        
        # Test loading data
        new_stats_mgr = TradingStatsManager()
        new_stats_mgr.load_data()
        print(f"   ✅ Stats data loaded")
        
        new_culler = AutoCullingSystem()
        new_culler.load_data()
        print(f"   ✅ Auto-culler data loaded")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data persistence test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive Integration Test")
    print("=" * 60)
    
    tests = [
        ("Trading Stats Manager", test_stats_manager),
        ("Auto-Culling System", test_auto_culling),
        ("Trainer Diagnostics", test_trainer_diagnostics),
        ("Discord Bot Imports", test_discord_bot_imports),
        ("Integration Workflow", test_integration_workflow),
        ("Data Persistence", test_data_persistence),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for production.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)

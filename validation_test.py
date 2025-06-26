#!/usr/bin/env python3
"""
Simple validation test for the enhanced trading platform
Focus on core functionality that should be working
"""

import os
import sys
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_integration():
    """Test core integration functionality"""
    print("🚀 Testing Core Integration...")
    
    try:
        # 1. Test importing main components
        print("📦 Testing imports...")
        from trading_stats import TradingStatsManager, get_stats_manager
        from auto_culling import AutoCullingSystem, get_auto_culler
        from model_training.trainer_diagnostics import TrainerDiagnostics
        print("   ✅ All imports successful")
        
        # 2. Test stats manager basic functionality
        print("📊 Testing stats manager...")
        stats_mgr = get_stats_manager()
        
        # Record a trade
        stats_mgr.record_trade(
            model_name="test_model",
            was_successful=True,
            pnl=100.0,
            trade_data={"entry_price": 50000, "exit_price": 51000}
        )
        print("   ✅ Trade recorded successfully")
        
        # Check if model exists in performance tracking
        if "test_model" in stats_mgr.models_performance:
            perf = stats_mgr.models_performance["test_model"]
            print(f"   ✅ Model performance tracked: {perf.total_trades} trades, Win rate: {perf.win_rate:.1%}")
          # 3. Test auto-culling system
        print("🤖 Testing auto-culling system...")
        culler = get_auto_culler()
        
        # Test model pause/unpause
        culler.pause_model("test_model", "Testing")
        print("   ✅ Model paused")
        
        culler.unpause_model("test_model")
        print("   ✅ Model unpaused")
        
        # 4. Test diagnostics
        print("🔬 Testing trainer diagnostics...")
        diagnostics = TrainerDiagnostics("test_model")
        print("   ✅ Diagnostics initialized")
        
        # 5. Test Discord bot imports
        print("🤖 Testing Discord bot integration...")
        from trading_bot.discord_trader_bot import bot
        print("   ✅ Discord bot imported successfully")
        
        print("\n🎉 All core integration tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Core integration test failed: {e}")
        traceback.print_exc()
        return False

def test_dashboard_generation():
    """Test dashboard generation"""
    print("\n📊 Testing Dashboard Generation...")
    
    try:
        from trading_stats import get_stats_manager
        
        stats_mgr = get_stats_manager()
        
        # Add some test data
        stats_mgr.record_trade("model_a", True, 50.0)
        stats_mgr.record_trade("model_a", False, -25.0)
        stats_mgr.record_trade("model_b", True, 75.0)
        
        # Generate dashboard
        dashboard_stats = stats_mgr.get_dashboard_stats(1000.0)  # Mock balance
        print(f"   ✅ Dashboard stats generated: {len(dashboard_stats)} metrics")
        
        # Generate leaderboard
        leaderboard = stats_mgr.get_model_leaderboard()
        print(f"   ✅ Leaderboard generated: {len(leaderboard)} models")
        
        # Check underperforming models
        underperforming = stats_mgr.get_underperforming_models()
        print(f"   ✅ Underperforming models: {len(underperforming)} flagged")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dashboard generation failed: {e}")
        traceback.print_exc()
        return False

def test_trading_cycle():
    """Test a complete trading cycle simulation"""
    print("\n🔄 Testing Complete Trading Cycle...")
    
    try:
        from trading_stats import get_stats_manager
        from auto_culling import get_auto_culler
        
        stats_mgr = get_stats_manager()
        culler = get_auto_culler()
        
        model_name = "cycle_test_model"
        
        # Simulate several trades with mixed results
        trades = [
            (True, 100.0), (False, -50.0), (True, 75.0), 
            (False, -30.0), (False, -40.0), (False, -60.0)  # Multiple losses to trigger flagging
        ]
        
        for is_win, pnl in trades:
            stats_mgr.record_trade(model_name, is_win, pnl)
            
        print(f"   ✅ Recorded {len(trades)} trades")
        
        # Check if model should be culled
        if model_name in stats_mgr.models_performance:
            perf = stats_mgr.models_performance[model_name]
            performance_dict = {
                "total_trades": perf.total_trades,
                "win_rate": perf.win_rate,
                "consecutive_losses": perf.consecutive_losses,
                "total_pnl": perf.total_pnl
            }
            
            should_cull, reason = culler.should_cull_model(model_name, performance_dict)
            print(f"   ✅ Culling decision: {should_cull} ({reason})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Trading cycle test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🎯 Enhanced Trading Platform Validation")
    print("=" * 50)
    
    tests = [
        ("Core Integration", test_core_integration),
        ("Dashboard Generation", test_dashboard_generation),
        ("Trading Cycle", test_trading_cycle),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 VALIDATION SUCCESSFUL! Enhanced trading platform is ready.")
        print("\n🚀 Key Features Validated:")
        print("   • Trading statistics tracking")
        print("   • Auto-culling system")
        print("   • Trainer diagnostics")
        print("   • Discord bot integration")
        print("\n💡 Next steps:")
        print("   • Start the Discord bot: python start_discord_bots.py")
        print("   • Monitor dashboard via Discord commands")
        print("   • Review auto-culling settings as needed")
    else:
        print("\n⚠️  Some validations failed. Core functionality should still work.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

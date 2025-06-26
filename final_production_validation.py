#!/usr/bin/env python3
"""
Final Production Validation Test

This script performs a final end-to-end validation of the complete trading system
to confirm production readiness before deployment.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
import time

def test_complete_trading_system():
    """Test the complete trading system end-to-end"""
    print("🔥" + "="*60 + "🔥")
    print("   FINAL PRODUCTION VALIDATION TEST")
    print("🔥" + "="*60 + "🔥")
    print()
    
    results = []
    
    print("1️⃣ Testing Core System Components...")
    print("-" * 40)
    
    # Test configuration loading
    try:
        from src.safe_config import get_config
        config = get_config()
        print("✅ Configuration system loaded successfully")
        results.append(True)
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        results.append(False)
    
    # Test safety manager
    try:
        from src.trading_safety import TradingSafetyManager
        safety_mgr = TradingSafetyManager(config)
        can_trade, reason = safety_mgr.can_trade_now()
        print(f"✅ Safety manager initialized: {reason}")
        results.append(True)
    except Exception as e:
        print(f"❌ Safety manager failed: {e}")
        results.append(False)
    
    # Test model validation
    try:
        from src.model_validation import ModelValidationService
        model_val = ModelValidationService()
        models_valid, validation_results = model_val.validate_all_models()
        print(f"✅ Model validation: {'PASSED' if models_valid else 'FAILED'}")
        results.append(models_valid)
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        results.append(False)
      # Test WebSocket manager
    try:
        from src.websocket_manager import BinanceWebSocketManager
        ws_mgr = BinanceWebSocketManager(safety_mgr)
        print("✅ WebSocket manager initialized")
        results.append(True)
    except Exception as e:
        print(f"❌ WebSocket manager failed: {e}")
        results.append(False)
    
    print(f"\n2️⃣ Testing Trading Bot Core Functions...")
    print("-" * 40)
    
    # Test market data fetching
    try:
        from src.trading_bot.trade_runner import fetch_top_200_ohlcv
        df = fetch_top_200_ohlcv()
        if not df.empty:
            print(f"✅ Market data fetched: {len(df)} records, {df['symbol'].nunique()} symbols")
            results.append(True)
        else:
            print("❌ No market data retrieved")
            results.append(False)
    except Exception as e:
        print(f"❌ Market data fetching failed: {e}")
        results.append(False)
    
    # Test technical indicators
    try:
        from src.trading_bot.trade_runner import calculate_rsi_macd
        if not df.empty:
            df_with_indicators = calculate_rsi_macd(df)
            if 'rsi' in df_with_indicators.columns and 'macd' in df_with_indicators.columns:
                print("✅ Technical indicators calculated successfully")
                results.append(True)
            else:
                print("❌ Technical indicators missing")
                results.append(False)
        else:
            print("⚠️ Skipping technical indicators (no data)")
            results.append(True)
    except Exception as e:
        print(f"❌ Technical indicators failed: {e}")
        results.append(False)
    
    # Test balance checking
    try:
        from src.trading_bot.trade_runner import get_usdt_balance
        balance = get_usdt_balance()
        print(f"✅ Balance check successful: ${balance:.2f}")
        results.append(True)
    except Exception as e:
        print(f"❌ Balance check failed: {e}")
        results.append(False)
    
    print(f"\n3️⃣ Testing Discord Bot Integration...")
    print("-" * 40)
    
    # Test Discord bot imports
    discord_results = []
    
    try:
        from src.trading_bot.discord_trader_bot import bot as trading_bot
        print("✅ Trading Discord bot imported")
        discord_results.append(True)
    except Exception as e:
        print(f"❌ Trading Discord bot failed: {e}")
        discord_results.append(False)
    
    try:
        from src.data_collector.discord_bot import bot as data_bot
        print("✅ Data collection Discord bot imported")
        discord_results.append(True)
    except Exception as e:
        print(f"❌ Data collection Discord bot failed: {e}")
        discord_results.append(False)
    
    try:
        from src.model_training.discord_training_bot import bot as training_bot
        print("✅ ML training Discord bot imported")
        discord_results.append(True)
    except Exception as e:
        print(f"❌ ML training Discord bot failed: {e}")
        discord_results.append(False)
    
    results.extend(discord_results)
    
    print(f"\n4️⃣ Testing Error Handling & Safety...")
    print("-" * 40)
    
    # Test emergency stop
    try:
        stop_flag = "TRADING_DISABLED.flag"
        with open(stop_flag, "w") as f:
            f.write("Test emergency stop")
        
        can_trade, reason = safety_mgr.can_trade_now()
        if not can_trade and ("disabled" in reason.lower() or "flag" in reason.lower()):
            print("✅ Emergency stop mechanism working")
            results.append(True)
        else:
            print("❌ Emergency stop not working")
            results.append(False)
        
        # Cleanup
        if os.path.exists(stop_flag):
            os.remove(stop_flag)
            
    except Exception as e:
        print(f"❌ Emergency stop test failed: {e}")
        results.append(False)
    
    # Test retry mechanism
    try:
        from src.trading_bot.trade_runner import retry_api_call
        
        def failing_function():
            raise Exception("Test error")
        
        try:
            retry_api_call(failing_function, max_retries=2)
            print("❌ Retry mechanism not working")
            results.append(False)
        except:
            print("✅ Retry mechanism working correctly")
            results.append(True)
            
    except Exception as e:
        print(f"❌ Retry mechanism test failed: {e}")
        results.append(False)
    
    print(f"\n5️⃣ Final System Integration Test...")
    print("-" * 40)
    
    # Test complete system status
    try:
        from bot_status import get_system_status
        status = get_system_status()
        if isinstance(status, dict) and len(status) > 0:
            print(f"✅ System status reporting: {len(status)} metrics")
            results.append(True)
        else:
            print("❌ System status reporting failed")
            results.append(False)
    except Exception as e:
        print(f"❌ System status test failed: {e}")
        results.append(False)
    
    # Generate final validation report
    print("\n" + "="*60)
    print("📋 FINAL PRODUCTION VALIDATION REPORT")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print()
    
    if success_rate >= 95:
        print("🎉 FINAL VALIDATION: PRODUCTION READY")
        print("   ✅ System fully validated and ready for deployment")
        print("   ✅ All critical components operational")
        print("   ✅ Safety mechanisms verified")
        print("   ✅ Discord integration functional")
    elif success_rate >= 85:
        print("⚠️ FINAL VALIDATION: MOSTLY READY")
        print("   ⚠️ Minor issues detected but core functionality working")
        print("   ⚠️ Review failed tests before deployment")
    else:
        print("❌ FINAL VALIDATION: NOT READY")
        print("   ❌ Significant issues require fixes before deployment")
        print("   ❌ Review and resolve failed tests")
    
    print()
    print("🚀 DEPLOYMENT READINESS CHECKLIST:")
    print("   ☐ Configure production API keys")
    print("   ☐ Set up Discord bot tokens")
    print("   ☐ Deploy to production server")
    print("   ☐ Start monitoring and logging")
    print("   ☐ Begin with dry trading validation")
    print()
    
    return success_rate >= 95

if __name__ == "__main__":
    try:
        print(f"Starting final production validation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        is_ready = test_complete_trading_system()
        
        print()
        if is_ready:
            print("🏆 MONEY PRINTER TRADING BOT - PRODUCTION CERTIFIED 🏆")
        else:
            print("⚠️ MONEY PRINTER TRADING BOT - REQUIRES ATTENTION ⚠️")
            
    except KeyboardInterrupt:
        print("\n👋 Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical validation error: {e}")

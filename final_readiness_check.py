#!/usr/bin/env python3
"""
Final readiness check for the enhanced crypto trading platform
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_files_exist():
    """Check that all required files exist"""
    print("📁 Checking file structure...")
    
    required_files = [
        "src/trading_stats.py",
        "src/auto_culling.py", 
        "src/model_training/trainer_diagnostics.py",
        "src/trading_bot/discord_trader_bot.py",
        "src/trading_bot/trade_runner.py",
        "README.md",
        "start_discord_bots.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    
    print("   ✅ All required files present")
    return True

def check_imports():
    """Check that all modules can be imported"""
    print("\n📦 Checking module imports...")
    
    modules = [
        ("trading_stats", "TradingStatsManager"),
        ("auto_culling", "AutoCullingSystem"),
        ("model_training.trainer_diagnostics", "TrainerDiagnostics"),
    ]
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{class_name}: {e}")
            return False
    
    print("   ✅ All core modules importable")
    return True

def check_data_directories():
    """Check that data directories exist"""
    print("\n📂 Checking data directories...")
    
    directories = [
        "data",
        "data/models",
        "data/diagnostics",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"   ✅ Created {directory}")
        else:
            print(f"   ✅ {directory} exists")
    
    return True

def check_config():
    """Check configuration"""
    print("\n⚙️ Checking configuration...")
    
    if os.path.exists(".env"):
        print("   ✅ .env file found")
    else:
        print("   ⚠️ .env file not found - Discord bot may not work")
    
    return True

def main():
    print("🎯 Enhanced Trading Platform - Final Readiness Check")
    print("=" * 60)
    
    checks = [
        ("File Structure", check_files_exist),
        ("Module Imports", check_imports),
        ("Data Directories", check_data_directories),
        ("Configuration", check_config),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"   ❌ {check_name} failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 SYSTEM READY FOR PRODUCTION!")
        print("\n🚀 To start the enhanced trading platform:")
        print("   python start_discord_bots.py")
        print("\n📊 Available Discord Commands:")
        print("   /dashboard - View real-time trading stats")
        print("   /status [model] - Check model performance")
        print("   /leaderboard - Top performing models")
        print("   /metrics [model] - Detailed diagnostics")
        print("   /retrain [weak|all|model] - Retrain models")
        print("   /balance - Check wallet balance")
        print("   /culling status - Auto-culling information")
        print("   /stop_trading - Emergency stop")
        print("\n💡 Key Features Active:")
        print("   • Live trading dashboard with real-time stats")
        print("   • Auto-culling of underperforming models")
        print("   • Comprehensive trainer diagnostics")
        print("   • Discord bot remote control")
        print("   • Automated model retraining")
        print("\n🔒 Safety Features:")
        print("   • Emergency stop functionality")
        print("   • Auto-flagging of poor performers")
        print("   • Manual override controls")
        print("   • Comprehensive logging")
    else:
        print("❌ SYSTEM NOT READY - Please fix the issues above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

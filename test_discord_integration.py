#!/usr/bin/env python3
"""
Test Discord Bot Integration
Quick test to validate Discord bot functionality with new Drive manager.
"""

import os
import sys
import traceback
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_discord_bot_imports():
    """Test Discord bot imports and basic functionality"""
    print("🤖 Testing Discord bot imports...")
    
    try:
        # Test importing the Discord bot
        from src.trading_bot.discord_trader_bot import bot
        print("✅ Discord bot module imported successfully")
        
        # Test if bot is properly configured
        if bot:
            print("✅ Bot instance created")
            print(f"   Command prefix: {bot.command_prefix}")
            print(f"   Commands loaded: {len(bot.commands)}")
            
            # List available commands
            command_names = [cmd.name for cmd in bot.commands]
            print(f"   Available commands: {', '.join(command_names)}")
        else:
            print("❌ Bot instance not created")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Discord bot test error: {e}")
        traceback.print_exc()
        return False

def test_drive_manager_in_bot():
    """Test Drive manager integration in Discord bot"""
    print("\n📁 Testing Drive manager integration in Discord bot...")
    
    try:
        from src.drive_manager import get_drive_manager
        
        # Test getting drive manager (same function used in Discord bot)
        manager = get_drive_manager()
        print("✅ Drive manager accessible from Discord bot context")
        
        # Test getting status (used in /drive_status command)
        status = manager.get_status()
        print(f"✅ Drive status accessible: enabled={status['enabled']}, authenticated={status['authenticated']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Drive manager integration test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run Discord bot integration tests"""
    print("🧪 Discord Bot Integration Test")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    tests = [
        ("Discord Bot Imports", test_discord_bot_imports),
        ("Drive Manager Integration", test_drive_manager_in_bot),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("=" * 60)
        
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ PASS {test_name}")
            else:
                print(f"❌ FAIL {test_name}")
        except Exception as e:
            print(f"❌ FAIL {test_name} - Error: {e}")
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if i < passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    success_rate = (passed / total) * 100
    print(f"📈 Results: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if passed == total:
        print("🎉 All Discord bot tests passed!")
        return True
    else:
        print("⚠️ Some Discord bot tests failed.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        traceback.print_exc()
        sys.exit(1)

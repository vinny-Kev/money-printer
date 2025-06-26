#!/usr/bin/env python3
"""
Individual Discord Bot Starter - Capture startup errors
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def test_bot_startup(bot_name, bot_path, working_dir=None):
    """Test individual bot startup and capture errors"""
    print(f"\n🔄 Testing {bot_name} startup...")
    print(f"Bot path: {bot_path}")
    if working_dir:
        print(f"Working dir: {working_dir}")
    
    original_dir = os.getcwd()
    
    try:
        # Change to working directory if specified
        if working_dir:
            os.chdir(working_dir)
        
        # Start the bot process
        process = subprocess.Popen(
            [sys.executable, bot_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1
        )
        
        # Give it a few seconds to start
        time.sleep(3)
        
        # Check if it's still running
        if process.poll() is None:
            print(f"✅ {bot_name} started successfully and is running")
            process.terminate()
            process.wait()
            return True
        else:
            # Bot stopped, get the output
            output, _ = process.communicate()
            print(f"❌ {bot_name} stopped with exit code: {process.returncode}")
            print("Error output:")
            print(output)
            return False
            
    except Exception as e:
        print(f"❌ Failed to start {bot_name}: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def main():
    print("="*60)
    print("🤖 INDIVIDUAL DISCORD BOT STARTUP TESTING")
    print("="*60)
    
    # Test each bot individually
    bots = [
        ("Trading Bot", "discord_trader_bot.py", "src/trading_bot"),
        ("Data Collector Bot", "discord_bot.py", "src/data_collector"),
        ("Training Bot", "discord_training_bot.py", "src/model_training"),
    ]
    
    results = []
    for bot_name, bot_file, working_dir in bots:
        success = test_bot_startup(bot_name, bot_file, working_dir)
        results.append((bot_name, success))
    
    print("\n" + "="*60)
    print("📊 STARTUP TEST RESULTS")
    print("="*60)
    
    passed = 0
    for name, success in results:
        status = "✅ STARTED" if success else "❌ FAILED"
        print(f"{name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} bots started successfully")

if __name__ == "__main__":
    main()

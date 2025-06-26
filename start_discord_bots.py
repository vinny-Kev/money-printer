#!/usr/bin/env python3
"""
Discord Bot Launcher - Start Discord bots for trading system
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_discord_config():
    """Check if Discord configuration is properly set up"""
    # Check essential Discord bot configuration
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    channel_id = os.getenv('DISCORD_CHANNEL_ID')
    
    missing_vars = []
    
    if not bot_token or bot_token == 'your_bot_token_here':
        missing_vars.append('DISCORD_BOT_TOKEN')
    
    if not channel_id or channel_id == 'your_channel_id_here':
        missing_vars.append('DISCORD_CHANNEL_ID')
    
    # DISCORD_USER_ID is optional - if not set, we'll note it but not fail
    user_id = os.getenv('DISCORD_USER_ID')
    if not user_id:
        print("⚠️  DISCORD_USER_ID not set - Trading bot commands will be available to all users")
        print("   Add DISCORD_USER_ID=your_user_id to .env for restricted access")
    
    if missing_vars:
        print("❌ Discord configuration incomplete!")
        print(f"Missing or placeholder values for: {', '.join(missing_vars)}")
        print("\n📋 Your current Discord settings:")
        print(f"   DISCORD_BOT_TOKEN: {'✅ Set' if bot_token else '❌ Missing'}")
        print(f"   DISCORD_CHANNEL_ID: {'✅ Set' if channel_id else '❌ Missing'}")
        print(f"   DISCORD_USER_ID: {'✅ Set' if user_id else '⚠️ Optional (not set)'}")
        return False
    
    print("✅ Discord configuration looks good!")
    print(f"   Bot Token: {bot_token[:20]}...")
    print(f"   Channel ID: {channel_id}")
    if user_id:
        print(f"   User ID: {user_id} (authorized user)")
    return True

def start_trading_bot():
    """Start the trading Discord bot"""
    print("🤖 Starting Trading Discord Bot...")
    trading_bot_path = Path("src/trading_bot/discord_trader_bot.py")
    
    if not trading_bot_path.exists():
        print(f"❌ Trading bot not found at {trading_bot_path}")
        return None
    
    try:
        # Change to the trading_bot directory and start the bot
        os.chdir("src/trading_bot")
        process = subprocess.Popen([sys.executable, "discord_trader_bot.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, 
                                 text=True, 
                                 bufsize=1)
        os.chdir("../..")  # Go back to root
        print("✅ Trading bot started!")
        return process
    except Exception as e:
        print(f"❌ Failed to start trading bot: {e}")
        os.chdir("../..")  # Ensure we're back at root
        return None

def start_training_bot():
    """Start the training Discord bot"""
    print("🧠 Starting Training Discord Bot...")
    training_bot_path = Path("src/model_training/discord_training_bot.py")
    
    if not training_bot_path.exists():
        print(f"❌ Training bot not found at {training_bot_path}")
        return None
    
    try:
        # Change to the model_training directory and start the bot
        os.chdir("src/model_training")
        process = subprocess.Popen([sys.executable, "discord_training_bot.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, 
                                 text=True, 
                                 bufsize=1)
        os.chdir("../..")  # Go back to root
        print("✅ Training bot started!")
        return process
    except Exception as e:
        print(f"❌ Failed to start training bot: {e}")
        os.chdir("../..")  # Ensure we're back at root
        return None

def start_data_collector_bot():
    """Start the data collector Discord bot"""
    print("📊 Starting Data Collector Discord Bot...")
    collector_bot_path = Path("src/data_collector/discord_bot.py")
    
    if not collector_bot_path.exists():
        print(f"❌ Data collector bot not found at {collector_bot_path}")
        return None
    
    try:
        # Change to the data_collector directory and start the bot
        os.chdir("src/data_collector")
        process = subprocess.Popen([sys.executable, "discord_bot.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, 
                                 text=True, 
                                 bufsize=1)
        os.chdir("../..")  # Go back to root
        print("✅ Data collector bot started!")
        return process
    except Exception as e:
        print(f"❌ Failed to start data collector bot: {e}")
        os.chdir("../..")  # Ensure we're back at root
        return None

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🤖 DISCORD BOT LAUNCHER")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check Discord configuration
    if not check_discord_config():
        print("\n⚠️  Configure Discord settings in .env file first!")
        return
    
    print("\n🚀 Available Discord Bots:")
    print("1. Trading Bot - Control trading operations")
    print("2. Data Collector Bot - Control data scraping")
    print("3. Training Bot - Control ML training pipeline")
    print("4. All Bots")
    
    choice = input("\nWhich bot would you like to start? (1/2/3/4): ").strip()
    
    processes = []
    
    if choice in ['1', '4']:
        trading_process = start_trading_bot()
        if trading_process:
            processes.append(('Trading Bot', trading_process))
    
    if choice in ['2', '4']:
        data_process = start_data_collector_bot()
        if data_process:
            processes.append(('Data Collector Bot', data_process))
    
    if choice in ['3', '4']:
        training_process = start_training_bot()
        if training_process:
            processes.append(('Training Bot', training_process))
    
    if not processes:
        print("❌ No bots were started successfully!")
        return
    
    print(f"\n✅ Started {len(processes)} Discord bot(s)")
    print("\n📱 Bot Commands Available:")
    print("Trading Bot:")
    print("  /start_dry_trade [count] - Start dry trading")
    print("  /start_live_trade [count] - Start live trading")
    print("  /balance - Check account balance")
    print("\nData Collector Bot:")
    print("  Use slash commands in Discord for data operations")
    print("\nTraining Bot:")
    print("  /train_rf - Train Random Forest model")
    print("  /train_xgb - Train XGBoost model")
    print("  /incremental - Incremental training")
    print("  /status - Check training status")
    print("  /model_info - View model information")
    
    print("\n🛑 Press Ctrl+C to stop all bots")
    
    try:
        # Monitor processes
        while True:
            time.sleep(1)
            # Check if any process has died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"❌ {name} has stopped!")
                    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping Discord bots...")
        for name, process in processes:
            print(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("✅ All Discord bots stopped!")

if __name__ == "__main__":
    main()

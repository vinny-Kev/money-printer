"""
Money Printer - Automated Crypto Trading System
Main entry point for the trading system with multiple modules.
Production Ready - Version 1.0.1 - Railway Deployment Optimized
Final Production Release: June 27, 2025
"""
import os
import sys
import argparse
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def run_data_collector():
    """Run the data collection module with proper signal handling."""
    print("🚀 Starting Data Collector...")
    print("💡 Press Ctrl+C to stop data collection")
    
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\n🛑 Data collection stopped by user")
        print("📊 Cleaning up and saving data...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        from data_collector.data_scraper import main
        main()
    except KeyboardInterrupt:
        print("\n🛑 Data collection interrupted")
        sys.exit(0)

def run_timed_data_collector(hours):
    """Run data collection for a specified number of hours."""
    import signal
    import sys
    import time
    import threading
    from datetime import datetime, timedelta
    
    print(f"🚀 Starting Timed Data Collector for {hours} hours...")
    print(f"⏰ Will stop automatically at: {(datetime.now() + timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 Press Ctrl+C to stop early")
    
    # Global flag for stopping
    stop_flag = threading.Event()
    
    def signal_handler(sig, frame):
        print("\n🛑 Data collection stopped by user")
        stop_flag.set()
        sys.exit(0)
    
    def timer_handler():
        """Timer function that stops collection after specified hours"""
        time.sleep(hours * 3600)  # Convert hours to seconds
        if not stop_flag.is_set():
            print(f"\n⏰ Timer expired! {hours} hours completed.")
            print("🛑 Stopping data collection gracefully...")
            stop_flag.set()
            # Send interrupt signal to main thread
            os.kill(os.getpid(), signal.SIGINT)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start timer thread
    timer_thread = threading.Thread(target=timer_handler, daemon=True)
    timer_thread.start()
    
    try:
        # Import and run the Railway data scraper
        import asyncio
        from src.data_collector.railway_data_scraper import main as railway_main
        asyncio.run(railway_main(stop_flag))
    except ImportError as ie:
        print(f"⚠️ Railway scraper import failed: {ie}")
        # Fallback to regular scraper
        try:
            from data_collector.data_scraper import main
            main()
        except KeyboardInterrupt:
            print("\n🛑 Data collection completed")
    except KeyboardInterrupt:
        print("\n🛑 Data collection completed")
    finally:
        stop_flag.set()
        print(f"📊 Data collection session finished after {hours} hours or manual stop")

def run_model_training(model_type="random_forest"):
    """Run model training."""
    print(f"🤖 Starting Model Training ({model_type})...")
    
    if model_type.lower() in ["rf", "random_forest"]:
        from model_training.random_forest_trainer import main
        main()
    elif model_type.lower() in ["xgb", "xgboost"]:
        from model_variants.xgboost_trainer import main
        main()
    else:
        print(f"❌ Unknown model type: {model_type}")
        print("Available types: random_forest, xgboost")

def run_trading_bot():
    """Run the trading bot."""
    print("💰 Starting Trading Bot...")
    from trading_bot.trade_runner import main
    main()

def run_discord_data_bot():
    """Run the Discord data collection bot."""
    print("🤖 Starting Discord Data Collection Bot...")
    from data_collector.discord_bot import main
    main()

def run_discord_trading_bot():
    """Run the Discord trading bot."""
    print("🤖 Starting Discord Trading Bot...")
    from src.lightweight_discord_bot import main
    main()

def run_production_bot():
    """Run the full production bot with Discord interface."""
    print("🚀 Starting Money Printer Production Bot...")
    print("🤖 Discord Bot Interface Active")
    print("💰 Trading System Ready")
    print("📊 Data Collection Ready") 
    print("🧠 Model Training Ready")
    
    # Start the lightweight Discord bot which handles everything
    from src.lightweight_discord_bot import main
    main()

def test_notifications():
    """Test Discord notification system."""
    print("🔔 Testing Discord Notifications...")
    from discord_notifications import (
        send_scraper_notification,
        send_trainer_notification,
        send_trader_notification,
        send_general_notification
    )
    
    # Test all notification types
    print("Testing all notification channels...")
    send_general_notification("🧪 **System Test**: Testing the centralized Discord notification system")
    send_scraper_notification("🧪 **Scraper Test**: Testing data scraper notifications")
    send_trainer_notification("🧪 **Trainer Test**: Testing model trainer notifications")
    send_trader_notification("🧪 **Trader Test**: Testing trading bot notifications")
    print("✅ Notification test completed! Check your Discord channels.")

def show_status():
    """Show system status including real Binance balance."""
    print("=== Money Printer System Status ===")
    
    # Check real Binance balance
    try:
        print("💰 Checking Real Binance Balance...")
        from binance.client import Client
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if api_key and secret_key:
            client = Client(api_key=api_key, api_secret=secret_key)
            account_info = client.get_account()
            
            # Get USDT balance
            usdt_balance = 0.0
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            print(f"💵 USDT Balance: {usdt_balance:.2f} USDT")
            print(f"📊 Account Type: {account_info.get('accountType', 'Unknown')}")
            print(f"🔄 Can Trade: {account_info.get('canTrade', False)}")
            
            # Check if balance is sufficient for trading
            if usdt_balance >= 5.0:
                print("✅ Balance sufficient for trading")
            else:
                print("⚠️ Balance may be insufficient for trading (recommended: $5+ USDT)")
        else:
            print("❌ Binance API credentials not configured")
    except Exception as e:
        print(f"❌ Error checking Binance balance: {e}")
    
    print("\n" + "="*50)
    
    # Check data availability
    try:
        from config import PARQUET_DATA_DIR, MODELS_DIR
        from data_collector.local_storage import list_parquet_files
        
        files = list_parquet_files()
        print(f"📊 Data Files: {len(files)} parquet files found")
        
        # Check models
        model_files = list(MODELS_DIR.rglob("*.pkl"))
        print(f"🤖 Models: {len(model_files)} model files found")
        
        # Show recent data
        if files:
            recent_files = sorted(files, key=lambda x: x['modified'], reverse=True)[:5]
            print("\nRecent Data Files:")
            for file_info in recent_files:
                print(f"  📄 {file_info['filename']} - {file_info['size']/1024:.1f} KB - {file_info['modified']}")
        else:
            print("\n⚠️ No data files found. Run data collection first.")
        
        # Show available models
        if model_files:
            print("\nAvailable Models:")
            for model_file in model_files:
                rel_path = model_file.relative_to(MODELS_DIR)
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"  🧠 {rel_path} - {size_mb:.1f} MB")
        else:
            print("\n⚠️ No trained models found. Run model training first.")
    except Exception as e:
        print(f"❌ Error checking data/models: {e}")
    
    print("\n" + "="*50)
    
    # Check environment configuration
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        live_trading = os.getenv('LIVE_TRADING', 'false').lower() == 'true'
        use_drive = os.getenv('USE_GOOGLE_DRIVE', 'false').lower() == 'true'
        discord_webhook = os.getenv('DISCORD_WEBHOOK', '')
        
        print("⚙️ Configuration Status:")
        print(f"  🔴 Live Trading: {'✅ ENABLED' if live_trading else '❌ DISABLED'}")
        print(f"  ☁️ Google Drive: {'✅ ENABLED' if use_drive else '❌ DISABLED'}")
        print(f"  🤖 Discord: {'✅ CONFIGURED' if discord_webhook else '❌ NOT CONFIGURED'}")
        
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")

def main():
    parser = argparse.ArgumentParser(description="Money Printer - Automated Crypto Trading System")
    parser.add_argument("command", nargs='?', default="production", choices=[
        "collect", "train", "trade", "discord-data", "discord-trade", "status", "test-notifications", "production"
    ], help="Command to run (default: production)")
    parser.add_argument("--model", default="random_forest", 
                       choices=["random_forest", "rf", "xgboost", "xgb"],
                       help="Model type for training")
    parser.add_argument("--hours", type=float, default=None,
                       help="Hours to run data collection (e.g., --hours 2.5)")
    
    args = parser.parse_args()
    
    try:
        if args.command == "collect":
            if args.hours:
                run_timed_data_collector(args.hours)
            else:
                run_data_collector()
        elif args.command == "train":
            run_model_training(args.model)
        elif args.command == "trade":
            run_trading_bot()
        elif args.command == "discord-data":
            run_discord_data_bot()
        elif args.command == "discord-trade":
            run_discord_trading_bot()
        elif args.command == "status":
            show_status()
        elif args.command == "test-notifications":
            test_notifications()
        elif args.command == "production":
            run_production_bot()
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

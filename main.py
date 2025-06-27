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
    """Run the data collection module."""
    print("🚀 Starting Data Collector...")
    from data_collector.data_scraper import main
    main()

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
    """Show system status."""
    print("=== Money Printer System Status ===")
    
    # Check data availability
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
    
    # Show available models
    if model_files:
        print("\nAvailable Models:")
        for model_file in model_files:
            rel_path = model_file.relative_to(MODELS_DIR)
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  🧠 {rel_path} - {size_mb:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Money Printer - Automated Crypto Trading System")
    parser.add_argument("command", nargs='?', default="production", choices=[
        "collect", "train", "trade", "discord-data", "discord-trade", "status", "test-notifications", "production"
    ], help="Command to run (default: production)")
    parser.add_argument("--model", default="random_forest", 
                       choices=["random_forest", "rf", "xgboost", "xgb"],
                       help="Model type for training")
    
    args = parser.parse_args()
    
    try:
        if args.command == "collect":
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

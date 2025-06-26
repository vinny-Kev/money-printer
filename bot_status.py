#!/usr/bin/env python3
"""
Production Trading Bot Status CLI

Provides comprehensive status monitoring and control interface.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.safe_config import get_config
from src.trading_safety import TradingSafetyManager
from src.websocket_manager import BinanceWebSocketManager
from src.model_validation import ModelValidationService

def print_header(title: str):
    """Print formatted header"""
    print("=" * 70)
    print(f"🤖 {title.center(60)} 🤖")
    print("=" * 70)

def print_section(title: str):
    """Print formatted section header"""
    print(f"\n📊 {title}")
    print("-" * 50)

def format_uptime(start_time: datetime) -> str:
    """Format uptime duration"""
    uptime = datetime.utcnow() - start_time
    days = uptime.days
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    try:
        config = get_config()
        safety_mgr = TradingSafetyManager(config)
        model_val = ModelValidationService()
        model_val.register_model("random_forest_v1")
        
        # Get all status information
        safety_status = safety_mgr.get_status_report()
        
        # Check models
        models_valid, model_results = model_val.validate_all_models()
        
        # Check WebSocket status
        ws_status = {"status": "Not initialized"}
        try:
            ws_mgr = BinanceWebSocketManager(safety_mgr)
            ws_status = ws_mgr.get_connection_status()
        except:
            pass
        
        # Check trading readiness
        can_trade, trade_reason = safety_mgr.can_trade_now()
        can_trade_models, model_reason = model_val.can_trade()
        
        overall_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "trading_mode": "LIVE" if config.live_trading else "DRY",
            "overall_health": "HEALTHY" if (can_trade and can_trade_models) else "ISSUES",
            "safety_manager": safety_status,
            "model_validation": {
                "models_valid": models_valid,
                "can_trade": can_trade_models,
                "reason": model_reason,
                "details": model_results
            },
            "websocket": ws_status,
            "config": config.get_trading_config()
        }
        
        return overall_status
        
    except Exception as e:
        return {"error": f"Failed to get status: {e}"}

def display_status():
    """Display comprehensive bot status"""
    print_header("PRODUCTION TRADING BOT STATUS")
    
    status = get_system_status()
    
    if "error" in status:
        print(f"❌ Error: {status['error']}")
        return
    
    # Overall Health
    print_section("SYSTEM HEALTH")
    health = status["overall_health"]
    health_emoji = "✅" if health == "HEALTHY" else "⚠️"
    print(f"{health_emoji} Overall Status: {health}")
    print(f"🕒 Last Check: {status['timestamp']}")
    print(f"🎯 Trading Mode: {status['trading_mode']}")
    
    # Safety Manager Status
    print_section("SAFETY MANAGER")
    safety = status["safety_manager"]
    bot_status_emoji = "🟢" if safety["bot_status"] == "ACTIVE" else "🔴"
    print(f"{bot_status_emoji} Bot Status: {safety['bot_status']}")
    print(f"⏱️ Uptime: {safety['uptime_hours']:.1f} hours")
    print(f"💰 Total P&L: ${safety['total_pnl']:.2f}")
    print(f"📈 Daily Trades: {safety['daily_trades']}")
    print(f"📊 Hourly Trades: {safety['hourly_trades']}")
    print(f"🔒 Active Trades: {safety['active_trades']}")
    print(f"⏸️ Locked Symbols: {safety['locked_symbols']}")
    
    if safety["websocket_failures"] > 0:
        print(f"⚠️ WebSocket Failures: {safety['websocket_failures']}")
    
    if safety["last_data_age_minutes"]:
        age = safety["last_data_age_minutes"]
        age_emoji = "🟢" if age < 5 else "🟡" if age < 15 else "🔴"
        print(f"{age_emoji} Last Data: {age:.1f} minutes ago")
    
    if safety["api_rate_limited"]:
        print("🚫 API Rate Limited: YES")
    
    # Model Validation Status
    print_section("MODEL VALIDATION")
    models = status["model_validation"]
    model_emoji = "✅" if models["models_valid"] else "❌"
    print(f"{model_emoji} Models Valid: {models['models_valid']}")
    print(f"🤖 Can Trade: {models['can_trade']}")
    if not models["can_trade"]:
        print(f"📝 Reason: {models['reason']}")
    
    for model_name, details in models["details"].items():
        print(f"\n🔍 {model_name}:")
        perf = details["performance"].get("current_performance", {})
        if perf:
            print(f"   📊 Win Rate: {perf['win_rate']:.1f}%")
            print(f"   💰 Avg Profit: {perf['avg_profit']:.2f}%")
            print(f"   📈 Total Trades: {perf['total_trades']}")
            print(f"   🎯 Confidence Correlation: {perf['confidence_correlation']:.3f}")
    
    # WebSocket Status
    print_section("WEBSOCKET STATUS")
    ws = status["websocket"]
    if "data_is_fresh" in ws:
        ws_emoji = "✅" if ws["data_is_fresh"] else "❌"
        print(f"{ws_emoji} Data Fresh: {ws['data_is_fresh']}")
        print(f"📡 Subscribed Symbols: {ws['subscribed_symbols']}")
        print(f"💹 Price Data: {ws['price_data_symbols']} symbols")
        print(f"📊 Kline Data: {ws['kline_data_symbols']} symbols")
        
        for conn_name, conn_status in ws.get("connections", {}).items():
            status_emoji = "🟢" if conn_status["is_connected"] else "🔴"
            print(f"   {status_emoji} {conn_name}: {'Connected' if conn_status['is_connected'] else 'Disconnected'}")
            if conn_status["total_reconnects"] > 0:
                print(f"      🔄 Reconnects: {conn_status['total_reconnects']}")
    else:
        print("📡 WebSocket: Not initialized")
    
    # Configuration
    print_section("CONFIGURATION")
    config_data = status["config"]
    print(f"📈 Max Daily Trades: {config_data['max_daily_trades']}")
    print(f"⏱️ Max Hourly Trades: {config_data['max_hourly_trades']}")
    print(f"🛡️ Max Loss %: {config_data['bot_max_loss_percent']}%")
    print(f"💰 Max Position %: {config_data['max_position_size_percent']}%")
    print(f"📊 Min Model Win Rate: {config_data['min_model_winrate']}%")
    print(f"🔄 WebSocket Timeout: {config_data['websocket_timeout_minutes']} min")
    print(f"💸 Profit Reinvestment: {config_data['enable_profit_reinvestment']}")

def enable_trading():
    """Enable trading by removing disable flag"""
    flag_file = "TRADING_DISABLED.flag"
    if os.path.exists(flag_file):
        os.remove(flag_file)
        print("✅ Trading enabled - disable flag removed")
    else:
        print("ℹ️ Trading was already enabled")

def disable_trading(reason: str = "Manual disable"):
    """Disable trading by creating disable flag"""
    flag_file = "TRADING_DISABLED.flag"
    with open(flag_file, 'w') as f:
        f.write(f"Trading disabled at {datetime.utcnow().isoformat()}: {reason}")
    print(f"🛑 Trading disabled: {reason}")

def show_recent_trades(limit: int = 10):
    """Show recent trade history"""
    print_header(f"RECENT TRADES (Last {limit})")
    
    try:
        trades_file = "data/transactions/random_forest_v1_trades.csv"
        if not os.path.exists(trades_file):
            print("📝 No trade history found")
            return
        
        import pandas as pd
        df = pd.read_csv(trades_file)
        
        if len(df) == 0:
            print("📝 No trades in history")
            return
        
        # Get recent trades
        recent = df.tail(limit)
        
        print(f"📊 Total Trades: {len(df)}")
        print(f"✅ Successful: {df['was_successful'].sum()}")
        print(f"❌ Failed: {(~df['was_successful']).sum()}")
        print(f"📈 Win Rate: {(df['was_successful'].mean() * 100):.1f}%")
        print(f"💰 Avg P&L: {df['pnl_percent'].mean():.2f}%")
        print(f"💸 Total P&L: {df['pnl_amount'].sum():.2f}")
        
        print("\n📋 Recent Trades:")
        print("-" * 80)
        
        for _, trade in recent.iterrows():
            timestamp = pd.to_datetime(trade['timestamp']).strftime('%m-%d %H:%M')
            success_emoji = "✅" if trade['was_successful'] else "❌"
            print(f"{success_emoji} {timestamp} | {trade['coin']} | {trade['pnl_percent']:+6.2f}% | {trade['trade_duration_secs']:6.1f}s")
        
    except Exception as e:
        print(f"❌ Error loading trade history: {e}")

def show_model_performance():
    """Show detailed model performance metrics"""
    print_header("MODEL PERFORMANCE ANALYSIS")
    
    try:
        model_val = ModelValidationService()
        model_val.register_model("random_forest_v1")
        
        validation_status = model_val.get_validation_status()
        
        for model_name, details in validation_status["validation_details"].items():
            print(f"\n🤖 Model: {model_name}")
            print(f"📝 Version: {details['model_version']}")
            print(f"✅ Valid: {details['model_valid']}")
            
            if details['model_age_hours']:
                print(f"⏱️ Age: {details['model_age_hours']:.1f} hours")
            
            perf = details.get('current_performance')
            if perf:
                print(f"📊 Win Rate: {perf['win_rate']:.1f}%")
                print(f"💰 Avg Profit: {perf['avg_profit']:.2f}%")
                print(f"📈 Total Trades: {perf['total_trades']}")
                print(f"🎯 Confidence Correlation: {perf['confidence_correlation']:.3f}")
            
            drift = details.get('drift_status', {})
            if drift:
                drift_emoji = "⚠️" if drift['has_drift'] else "✅"
                print(f"{drift_emoji} Drift Status: {drift['reason']}")
        
    except Exception as e:
        print(f"❌ Error loading model performance: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Production Trading Bot Status & Control")
    parser.add_argument("command", nargs="?", default="status", 
                       choices=["status", "enable", "disable", "trades", "models"],
                       help="Command to execute")
    parser.add_argument("--reason", default="Manual disable", 
                       help="Reason for disabling trading")
    parser.add_argument("--limit", type=int, default=10,
                       help="Number of recent trades to show")
    
    args = parser.parse_args()
    
    if args.command == "status":
        display_status()
    elif args.command == "enable":
        enable_trading()
    elif args.command == "disable":
        disable_trading(args.reason)
    elif args.command == "trades":
        show_recent_trades(args.limit)
    elif args.command == "models":
        show_model_performance()

if __name__ == "__main__":
    main()

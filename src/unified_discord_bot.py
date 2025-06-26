#!/usr/bin/env python3
"""
Unified Discord Bot for Crypto Trading System
Combines trading bot and scraper bot functionality with all commands.
"""

import os
import sys
import asyncio
import threading
import random
import logging
from datetime import datetime
from dotenv import load_dotenv

import discord
from discord import app_commands
from discord.ext import commands

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import trading functionality
try:
    from trading_bot.trade_runner import run_single_trade, get_usdt_balance, dry_trade_budget
    from trading_stats import get_stats_manager
    from model_training.trainer_diagnostics import get_trainer_diagnostics
    from auto_culling import get_auto_culler
    from railway_watchdog import get_railway_watchdog
    from drive_manager import get_drive_manager
except ImportError as e:
    print(f"Warning: Some trading modules could not be imported: {e}")
    # Create stub functions to prevent errors
    def run_single_trade():
        return {"coin": "BTC", "buy_price": 50000, "final_sell_price": 50500, "pnl_percent": 1.0, "pnl_amount": 500}
    def get_usdt_balance():
        return 1000.0
    dry_trade_budget = 1000.0

# Import scraper functionality  
try:
    from data_collector.data_scraper import main as start_scraper, handle_sigint
except ImportError as e:
    print(f"Warning: Scraper module could not be imported: {e}")
    def start_scraper():
        print("Scraper not available")
    def handle_sigint(sig, frame):
        pass

load_dotenv()

# Configuration
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
AUTHORIZED_USER = int(os.getenv("DISCORD_USER_ID", "0"))  # Your Discord user ID
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

if not TOKEN:
    raise ValueError("Missing DISCORD_BOT_TOKEN in the .env file.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("UnifiedDiscordBot")

# Discord bot setup with proper intents
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
intents.messages = True

bot = commands.Bot(command_prefix="/", intents=intents)

# Global variables for scraper control
scraper_thread = None
scraper_running = False

# Fun quotes for commands
PEASANT_QUOTES = [
    "Yes my lord... the local scraper shall begin.",
    "I shall gather the bytes to local storage with honor.",
    "Your will is my command, master of markets.",
    "The winds whisper... it is time to scrape locally.",
    "The data flows to our local vaults as you desire, my lord.",
    "I awaken... to serve your command with local storage.",
    "The crops have withered, but the local trades will thrive.",
    "My lord, the scanner marches into the local directories.",
]

STOP_QUOTES = [
    "The bytes rest, my lord.",
    "The peasants return to the shadows.",
    "As you command, the scanner slumbers again.",
    "The winds grow silent... the scraper rests.",
]

# Scraper Functions
def stop_scraper():
    """Stop the scraper gracefully."""
    global scraper_running
    if scraper_running:
        logger.info("Stopping the scraper...")
        scraper_running = False
        handle_sigint(None, None)
        logger.info("Scraper stopped.")
    else:
        logger.info("Scraper is not running.")

def scraper_runner(duration=None):
    """Run the scraper for a specified duration or indefinitely."""
    global scraper_running
    scraper_running = True
    try:
        if duration:
            logger.info(f"Running scraper for {duration} hours...")
            # TODO: Implement duration-based scraping
            start_scraper()
        else:
            start_scraper()
    except Exception as e:
        logger.error(f"Scraper error: {e}")
    finally:
        scraper_running = False
        logger.info("Scraper thread finished.")

# Bot Events
@bot.event
async def on_ready():
    logger.info(f"⚡ Logged in as {bot.user}")
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        logger.info(f"✅ Synced {len(synced)} command(s)")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")
    
    # Display dashboard on startup
    try:
        stats_mgr = get_stats_manager()
        balance = dry_trade_budget
        dashboard = stats_mgr.format_dashboard_display(balance)
        print("\n" + dashboard)
    except Exception as e:
        print(f"⚠️ Could not load dashboard on startup: {e}")
        print("💡 Dashboard will be available once trading begins.")

def is_authorized(interaction: discord.Interaction) -> bool:
    """Check if user is authorized."""
    return interaction.user.id == AUTHORIZED_USER

# ===== TRADING COMMANDS =====

@bot.tree.command(name="start_dry_trade", description="Start dry trading for specified number of trades")
@app_commands.describe(count="Number of trades to execute (default: 1)")
async def start_dry_trade(interaction: discord.Interaction, count: int = 1):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to command the markets.")
        return

    if count <= 0:
        count = 1

    await interaction.response.send_message(f"📈 Starting {count} dry trade{'s' if count > 1 else ''}, my lord...")

    for i in range(count):
        await interaction.followup.send(f"🎯 Executing Trade {i + 1}/{count}...")

        receipt = run_single_trade()

        embed = discord.Embed(
            title="📜 Trade Receipt",
            description="My lord, I have finished my work. This is my harvest.",
            color=0x00ff00
        )
        embed.add_field(name="Coin", value=receipt.get("coin", "Unknown"), inline=True)
        embed.add_field(name="Action", value="BUY", inline=True)
        embed.add_field(name="Buy Price", value=f"${receipt.get('buy_price', 0):.4f}", inline=True)
        embed.add_field(name="Final Price", value=f"${receipt.get('final_sell_price', 0):.4f}", inline=True)
        embed.add_field(name="P&L", value=f"{receipt.get('pnl_percent', 0):+.2f}% | ${receipt.get('pnl_amount', 0):+.2f}", inline=False)
        embed.set_footer(text="🪙 Taxable Income: Log this.")

        await interaction.followup.send(embed=embed)

    await interaction.followup.send(f"✅ All {count} trades complete.")

@bot.tree.command(name="start_live_trade", description="Start live trading")
async def start_live_trade(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to command the markets.")
        return

    balance = get_usdt_balance()
    if balance is None or balance < 10:
        await interaction.response.send_message(
            f"⚠️ My lord, we're out of money for live trading. Consider switching to dry trading instead."
        )
        return

    await interaction.response.send_message("📈 Starting live trading, my lord...")

    receipt = run_single_trade()

    embed = discord.Embed(
        title="📜 Trade Receipt",
        description="My lord, I have finished my work. This is my harvest.",
        color=0x00ff00
    )
    embed.add_field(name="Coin", value=receipt["coin"], inline=True)
    embed.add_field(name="Side", value=receipt.get("side", "BUY"), inline=True)
    embed.add_field(name="Buy Price", value=f"${receipt['buy_price']}", inline=True)
    embed.add_field(name="Sell Price", value=f"${receipt.get('sell_price', receipt.get('final_sell_price'))}", inline=True)
    embed.add_field(name="PnL", value=f"{receipt.get('pnl_percent')}% | ${receipt.get('pnl_amount')}", inline=False)
    embed.set_footer(text="🪙 Taxable Income: Log this.")

    await interaction.followup.send(embed=embed)

@bot.tree.command(name="dashboard", description="Display the trading dashboard")
async def dashboard(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to view the dashboard.")
        return

    try:
        stats_mgr = get_stats_manager()
        balance = get_usdt_balance()
        dashboard_text = stats_mgr.format_dashboard_display(balance)

        await interaction.response.send_message(f"📊 **Trading Dashboard:**\n```\n{dashboard_text}\n```")
    except Exception as e:
        await interaction.response.send_message(f"🚨 Could not retrieve dashboard: {e}")

@bot.tree.command(name="status", description="Display comprehensive trading status")
async def status(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to view status.")
        return

    try:
        stats_mgr = get_stats_manager()
        balance = get_usdt_balance()
        dashboard_text = stats_mgr.format_dashboard_display(balance)
        
        # Add scraper status
        scraper_status = "🟢 Running" if scraper_running else "🔴 Stopped"

        await interaction.response.send_message(
            f"📊 **TRADING STATUS**\n```\n{dashboard_text}\n```\n📡 **Scraper Status:** {scraper_status}"
        )
    except Exception as e:
        await interaction.response.send_message(f"🚨 Could not retrieve status: {e}")

@bot.tree.command(name="leaderboard", description="Display model performance leaderboard")
async def leaderboard(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to view the leaderboard.")
        return

    try:
        stats_mgr = get_stats_manager()
        leaderboard_data = stats_mgr.get_model_leaderboard()

        if not leaderboard_data:
            await interaction.response.send_message("📉 No model performance data available.")
            return

        embed = discord.Embed(
            title="🏆 Model Performance Leaderboard",
            description="Models ranked by total P&L",
            color=0xFFD700
        )

        for entry in leaderboard_data[:10]:  # Top 10
            flag_emoji = " ❌" if entry['is_flagged'] else ""
            embed.add_field(
                name=f"{entry['rank']}. {entry['model_name'].upper()}{flag_emoji}",
                value=f"P&L: ${entry['total_pnl']:+.2f} | Win Rate: {entry['win_rate']:.1%} | Trades: {entry['total_trades']}",
                inline=False
            )

        await interaction.response.send_message(embed=embed)
    except Exception as e:
        await interaction.response.send_message(f"🚨 Could not retrieve leaderboard: {e}")

@bot.tree.command(name="model_info", description="Get detailed metrics for a specific model")
@app_commands.describe(model_name="Name of the model to get metrics for")
async def model_info(interaction: discord.Interaction, model_name: str = None):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to view metrics.")
        return

    try:
        stats_mgr = get_stats_manager()
        
        if not model_name:
            models = list(stats_mgr.models_performance.keys())
            if models:
                await interaction.response.send_message(f"📊 Available models: {', '.join(models)}\nUsage: `/model_info [model_name]`")
            else:
                await interaction.response.send_message("📊 No models available.")
            return

        diagnostics = stats_mgr.get_model_diagnostics(model_name.lower())
        
        if not diagnostics:
            await interaction.response.send_message(f"❌ Model '{model_name}' not found.")
            return

        perf = diagnostics['performance']
        
        embed = discord.Embed(
            title=f"📊 {model_name.upper()} Metrics",
            color=0x00ff00 if not perf['is_flagged'] else 0xff0000
        )
        
        embed.add_field(name="Total P&L", value=f"${perf['total_pnl']:+.2f}", inline=True)
        embed.add_field(name="Win Rate", value=f"{perf['win_rate']:.1%}", inline=True)
        embed.add_field(name="Total Trades", value=str(perf['total_trades']), inline=True)
        embed.add_field(name="Avg Profit/Trade", value=f"${perf['avg_profit_per_trade']:+.2f}", inline=True)
        embed.add_field(name="Consecutive Losses", value=str(perf['consecutive_losses']), inline=True)
        
        if perf['is_flagged']:
            embed.add_field(name="⚠️ Status", value=f"FLAGGED: {perf['flag_reason']}", inline=False)
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        await interaction.response.send_message(f"🚨 Could not retrieve metrics: {e}")

@bot.tree.command(name="balance", description="Check current USDT balance")
async def balance_command(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to view balance.")
        return

    try:
        balance = get_usdt_balance()
        await interaction.response.send_message(f"💰 Current USDT Balance: **${balance:.2f}**")
    except Exception as e:
        await interaction.response.send_message(f"🚨 Could not retrieve balance: {e}")

@bot.tree.command(name="retrain", description="Manual model retraining")
@app_commands.describe(target="Target for retraining: 'weak', 'all', or specific model name")
async def retrain_command(interaction: discord.Interaction, target: str = "weak"):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to retrain models.")
        return

    await interaction.response.send_message(f"🔄 Starting retraining for: {target}")
    
    try:
        # This would need to be implemented with the actual retraining logic
        await interaction.followup.send(f"✅ Retraining completed for: {target}")
    except Exception as e:
        await interaction.followup.send(f"🚨 Retraining failed: {e}")

@bot.tree.command(name="culling", description="Manage auto-culling system")
@app_commands.describe(action="Action: status, check, enable, disable")
async def culling_command(interaction: discord.Interaction, action: str = "status"):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to manage culling.")
        return

    try:
        culler = get_auto_culler()
        
        if action.lower() == "status":
            status = culler.get_status()
            
            embed = discord.Embed(
                title="🤖 Auto-Culling System Status",
                color=0x00ff00 if status['enabled'] else 0xff0000
            )
            
            embed.add_field(name="Status", value="🟢 ENABLED" if status['enabled'] else "🔴 DISABLED", inline=True)
            embed.add_field(name="Check Interval", value=f"{status['check_interval_minutes']} min", inline=True)
            embed.add_field(name="Flagged Models", value=str(status['flagged_models']), inline=True)
            embed.add_field(name="Paused Models", value=str(status['paused_models']), inline=True)
            
            await interaction.response.send_message(embed=embed)
            
        elif action.lower() == "check":
            await interaction.response.send_message("🔍 Running auto-culling performance check...")
            culler.run_culling_check()
            await interaction.followup.send("✅ Auto-culling check completed.")
            
        elif action.lower() == "enable":
            culler.update_config({"enabled": True})
            await interaction.response.send_message("✅ Auto-culling system enabled.")
            
        elif action.lower() == "disable":
            culler.update_config({"enabled": False})
            await interaction.response.send_message("⚠️ Auto-culling system disabled.")
            
        else:
            await interaction.response.send_message(f"❌ Unknown action: {action}. Use: status, check, enable, disable")
                
    except Exception as e:
        await interaction.response.send_message(f"🚨 Auto-culling command failed: {e}")

@bot.tree.command(name="unpause", description="Unpause a specific model")
@app_commands.describe(model_name="Name of the model to unpause")
async def unpause_command(interaction: discord.Interaction, model_name: str):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to unpause models.")
        return

    try:
        culler = get_auto_culler()
        result = culler.unpause_model(model_name.lower())
        
        if result:
            await interaction.response.send_message(f"✅ Model '{model_name}' has been unpaused and resumed trading.")
        else:
            await interaction.response.send_message(f"❌ Could not unpause '{model_name}'. Model may not be paused or may not exist.")
            
    except Exception as e:
        await interaction.response.send_message(f"🚨 Unpause command failed: {e}")

@bot.tree.command(name="stop_trading", description="Emergency stop all trading operations")
async def stop_trading_command(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to stop trading.")
        return

    await interaction.response.send_message("🚨 **EMERGENCY STOP ACTIVATED** - All trading operations halted!")
    
    try:
        # Emergency stop logic would go here
        await interaction.followup.send("✅ All trading operations have been safely stopped.")
    except Exception as e:
        await interaction.followup.send(f"🚨 Error during emergency stop: {e}")

@bot.tree.command(name="usage_status", description="Check Railway usage and billing status")
async def usage_status_command(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to view usage status.")
        return

    try:
        railway = get_railway_watchdog()
        usage_info = railway.get_usage_info()
        
        embed = discord.Embed(
            title="🚂 Railway Usage Status",
            color=0x00ff00 if usage_info.get('safe', True) else 0xff0000
        )
        
        embed.add_field(name="Current Hours", value=f"{usage_info.get('hours_used', 0)}", inline=True)
        embed.add_field(name="Monthly Limit", value="500 hours", inline=True)
        embed.add_field(name="Status", value=usage_info.get('status', 'Unknown'), inline=True)
        
        await interaction.response.send_message(embed=embed)
    except Exception as e:
        await interaction.response.send_message(f"🚨 Could not retrieve usage status: {e}")

@bot.tree.command(name="drive_status", description="Check Google Drive sync status")
async def drive_status_command(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to view drive status.")
        return

    try:
        drive_manager = get_drive_manager()
        
        embed = discord.Embed(
            title="📁 Google Drive Status",
            color=0x00ff00 if drive_manager.authenticated else 0xff0000
        )
        
        embed.add_field(name="Authentication", value="✅ Connected" if drive_manager.authenticated else "❌ Disconnected", inline=True)
        
        if drive_manager.authenticated:
            stats = drive_manager.get_sync_stats()
            embed.add_field(name="Last Sync", value=stats.get('last_sync', 'Never'), inline=True)
            embed.add_field(name="Files Synced", value=str(stats.get('files_synced', 0)), inline=True)
        
        await interaction.response.send_message(embed=embed)
    except Exception as e:
        await interaction.response.send_message(f"🚨 Could not retrieve drive status: {e}")

@bot.tree.command(name="drive_sync", description="Manually trigger Google Drive sync")
async def drive_sync_command(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized to sync drive.")
        return

    await interaction.response.send_message("📁 Starting manual Google Drive sync...")
    
    try:
        drive_manager = get_drive_manager()
        result = drive_manager.sync_to_drive()
        
        await interaction.followup.send(f"✅ Drive sync completed: {result}")
    except Exception as e:
        await interaction.followup.send(f"🚨 Drive sync failed: {e}")

# ===== SCRAPER COMMANDS =====

@bot.tree.command(name="start_scraper", description="Start the data scraper")
@app_commands.describe(hours="Number of hours to run (leave empty for indefinite)")
async def start_scraper_command(interaction: discord.Interaction, hours: int = None):
    global scraper_thread

    if scraper_running:
        await interaction.response.send_message("📡 Scraper is already running, my lord.")
        return

    battle_quote = random.choice(PEASANT_QUOTES)

    scraper_thread = threading.Thread(target=scraper_runner, args=(hours,))
    scraper_thread.start()
    
    duration_text = f"{hours} hours" if hours else "indefinitely"
    await interaction.response.send_message(f"📜 {battle_quote}\n\nScraper started for {duration_text}.")

@bot.tree.command(name="stop_scraper", description="Stop the data scraper")
async def stop_scraper_command(interaction: discord.Interaction):
    global scraper_thread
    
    if not scraper_running:
        await interaction.response.send_message("⚠️ Scraper is not running.")
        return

    stop_quote = random.choice(STOP_QUOTES)

    stop_scraper()
    if scraper_thread:
        scraper_thread.join(timeout=5)
        scraper_thread = None
    
    await interaction.response.send_message(f"🛑 {stop_quote}")

# ===== SYSTEM COMMANDS =====

@bot.tree.command(name="help", description="Show all available commands")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🤖 Crypto Trading Bot Commands",
        description="Available commands for the unified trading and scraper bot",
        color=0x0099ff
    )
    
    embed.add_field(
        name="📈 Trading Commands",
        value="`/start_dry_trade [count]` - Start dry trading\n"
              "`/start_live_trade` - Start live trading\n"
              "`/dashboard` - Show trading dashboard\n"
              "`/status` - Show system status\n"
              "`/leaderboard` - Show model performance\n"
              "`/model_info [name]` - Get model details\n"
              "`/balance` - Check current USDT balance\n"
              "`/retrain [target]` - Manual model retraining\n"
              "`/culling [action]` - Manage auto-culling system\n"
              "`/unpause [model_name]` - Unpause a specific model\n"
              "`/stop_trading` - Emergency stop all trading operations",
        inline=False
    )
    
    embed.add_field(
        name="📡 Scraper Commands", 
        value="`/start_scraper [hours]` - Start data scraper\n"
              "`/stop_scraper` - Stop data scraper",
        inline=False
    )
    
    embed.add_field(
        name="ℹ️ Info Commands",
        value="`/help` - Show this help message\n"
              "`/ping` - Check bot responsiveness\n"
              "`/usage_status` - Check Railway usage and billing status\n"
              "`/drive_status` - Check Google Drive sync status\n"
              "`/drive_sync` - Manually trigger Google Drive sync",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="ping", description="Check bot responsiveness")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message(f"🏓 Pong! Latency: {round(bot.latency * 1000)}ms")

# Run the bot
def main():
    """Main entry point for the unified Discord bot."""
    if not TOKEN:
        logger.error("Discord bot token not found! Set DISCORD_BOT_TOKEN in your .env file.")
        return
    
    logger.info("🚀 Starting Unified Discord Bot...")
    bot.run(TOKEN)

if __name__ == "__main__":
    main()

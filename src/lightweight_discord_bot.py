#!/usr/bin/env python3
"""
Enhanced Lightweight Discord Bot for Railway deployment
Includes scraping, training, and trading commands - Railway optimized
"""

import os
import sys
import logging
import asyncio
import threading
import random
from datetime import datetime
from dotenv import load_dotenv
from aiohttp import web

import discord
from discord import app_commands
from discord.ext import commands

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

load_dotenv()

# Configuration
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
AUTHORIZED_USER = int(os.getenv("DISCORD_USER_ID", "0"))
PORT = int(os.getenv("PORT", "8000"))  # Railway sets this automatically

if not TOKEN:
    raise ValueError("Missing DISCORD_BOT_TOKEN environment variable")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("EnhancedDiscordBot")

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# Global variables for system control
scraper_thread = None
scraper_running = False
trading_active = False

# Fun quotes for commands
PEASANT_QUOTES = [
    "Yes my lord... the scraper shall begin its work.",
    "I shall gather the market data to serve your trading empire.",
    "Your will is my command, master of automated profits.",
    "The data flows to our vaults as you desire, my lord.",
    "I awaken the trading algorithms to serve your command.",
    "The market scanner marches into action.",
]

STOP_QUOTES = [
    "The data gathering rests, my lord.",
    "The trading algorithms return to slumber.",
    "As you command, the systems rest.",
    "The market winds grow silent...",
]

# Import trading functionality with Railway-safe fallbacks
try:
    sys.path.append('/app')  # Ensure app directory is in path
    sys.path.append('/app/src')  # Ensure src directory is in path
    from trading_bot.trade_runner import run_single_trade, get_usdt_balance
    dry_trade_budget = 1000.0  # Default budget
    TRADING_AVAILABLE = True
    logger.info("✅ Trading modules loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Trading modules not available: {e}")
    TRADING_AVAILABLE = False
    # Create stub functions to prevent errors
    def run_single_trade():
        return {"coin": "BTC", "buy_price": 50000, "final_sell_price": 50500, "pnl_percent": 1.0, "pnl_amount": 500}
    def get_usdt_balance():
        return 1000.0
    dry_trade_budget = 1000.0

# Import trading stats with fallback
try:
    from trading_stats import get_stats_manager
    TRADING_STATS_AVAILABLE = True
    logger.info("✅ Trading stats module loaded")
except ImportError as e:
    logger.warning(f"⚠️ Trading stats not available: {e}")
    TRADING_STATS_AVAILABLE = False
    def get_stats_manager():
        class DummyStatsManager:
            def get_summary_stats(self):
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'best_trade': 0,
                    'worst_trade': 0,
                    'avg_trade': 0
                }
        return DummyStatsManager()

# Import scraper functionality with Railway-safe fallbacks
try:
    from data_collector.data_scraper import main as start_scraper
    SCRAPER_AVAILABLE = True
    logger.info("✅ Scraper modules loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Scraper modules not available: {e}")
    SCRAPER_AVAILABLE = False
    def start_scraper():
        logger.warning("Scraper not available in this deployment")
        return False

# Import model training with Railway-safe fallbacks
try:
    from model_training.random_forest_trainer import main as train_rf_model
    MODEL_TRAINING_AVAILABLE = True
    logger.info("✅ Model training modules loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Model training modules not available: {e}")
    MODEL_TRAINING_AVAILABLE = False
    def train_rf_model():
        logger.warning("Model training not available in this deployment")
        return False

def is_authorized(interaction: discord.Interaction) -> bool:
    """Check if user is authorized."""
    return interaction.user.id == AUTHORIZED_USER

def stop_scraper():
    """Stop the scraper gracefully."""
    global scraper_running
    if scraper_running:
        scraper_running = False
        logger.info("Stopping scraper...")
        # No need to call handle_sigint if scraper is basic
        return True
    return False

def run_scraper_thread():
    """Run scraper in background thread."""
    global scraper_running
    try:
        scraper_running = True
        logger.info("Starting scraper thread...")
        start_scraper()
    except Exception as e:
        logger.error(f"Scraper error: {e}")
    finally:
        scraper_running = False
        logger.info("Scraper thread ended")

async def start_background_scraper():
    """Start scraper in background."""
    global scraper_thread, scraper_running
    
    if scraper_running:
        return False, "Scraper is already running"
    
    try:
        scraper_thread = threading.Thread(target=run_scraper_thread, daemon=True)
        scraper_thread.start()
        return True, "Scraper started successfully"
    except Exception as e:
        return False, f"Failed to start scraper: {e}"

def is_authorized(interaction: discord.Interaction) -> bool:
    """Check if user is authorized."""
    return interaction.user.id == AUTHORIZED_USER

@bot.event
async def on_ready():
    logger.info(f"⚡ Enhanced Discord Bot logged in as {bot.user}")
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        logger.info(f"✅ Synced {len(synced)} command(s)")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

@bot.tree.command(name="ping", description="Check bot responsiveness")
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)
    await interaction.response.send_message(f"🏓 Pong! Latency: {latency}ms")

@bot.tree.command(name="status", description="Check comprehensive system status")
async def status(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized.")
        return
    
    embed = discord.Embed(
        title="🤖 Enhanced Trading Bot Status",
        description="Railway-Deployed Trading System",
        color=0x00ff00
    )
    
    # Bot status
    embed.add_field(name="🟢 Bot Status", value="Online & Ready", inline=True)
    embed.add_field(name="⚡ Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="🌐 Environment", value="Railway Cloud", inline=True)
    
    # System availability
    availability = []
    if TRADING_AVAILABLE:
        availability.append("✅ Trading")
    else:
        availability.append("❌ Trading")
    
    if SCRAPER_AVAILABLE:
        availability.append("✅ Data Scraping")
    else:
        availability.append("❌ Data Scraping")
    
    if MODEL_TRAINING_AVAILABLE:
        availability.append("✅ Model Training")
    else:
        availability.append("❌ Model Training")
    
    embed.add_field(name="🔧 Available Systems", value="\n".join(availability), inline=False)
    
    # Current operations
    operations = []
    if scraper_running:
        operations.append("🔄 Data Scraper: Running")
    else:
        operations.append("⏸️ Data Scraper: Stopped")
    
    if trading_active:
        operations.append("💰 Trading: Active")
    else:
        operations.append("⏸️ Trading: Inactive")
    
    embed.add_field(name="🔄 Current Operations", value="\n".join(operations), inline=False)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="start_scraper", description="Start the cryptocurrency data scraper")
async def start_scraper_command(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized.")
        return
    
    if not SCRAPER_AVAILABLE:
        await interaction.response.send_message("❌ **Data scraper not available** in this deployment.\nThis may be a lightweight version.")
        return
    
    # Defer response as scraper startup may take time
    await interaction.response.defer()
    
    success, message = await start_background_scraper()
    
    if success:
        quote = random.choice(PEASANT_QUOTES)
        embed = discord.Embed(
            title="🚀 Data Scraper Started",
            description=f"*{quote}*",
            color=0x00ff00
        )
        embed.add_field(name="Status", value="✅ Scraper is now running", inline=True)
        embed.add_field(name="Data Collection", value="📊 Gathering market data in background", inline=True)
        embed.add_field(name="Next Steps", value="Use `/status` to monitor progress", inline=False)
    else:
        embed = discord.Embed(
            title="❌ Scraper Start Failed",
            description=message,
            color=0xff0000
        )
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="stop_scraper", description="Stop the data scraper")
async def stop_scraper_command(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized.")
        return
    
    success = stop_scraper()
    quote = random.choice(STOP_QUOTES)
    
    if success:
        embed = discord.Embed(
            title="🛑 Data Scraper Stopped",
            description=f"*{quote}*",
            color=0xff9900
        )
        embed.add_field(name="Status", value="⏸️ Scraper has been stopped", inline=True)
    else:
        embed = discord.Embed(
            title="ℹ️ Scraper Not Running",
            description="The data scraper was not currently active.",
            color=0x0099ff
        )
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="start_dry_trade", description="Start dry trading (paper trading)")
@app_commands.describe(num_trades="Number of trades to execute (1-10)")
async def start_dry_trade(interaction: discord.Interaction, num_trades: int = 1):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized.")
        return
    
    if not TRADING_AVAILABLE:
        await interaction.response.send_message("❌ **Trading system not available** in this deployment.\nThis may be a lightweight version.")
        return
    
    if num_trades < 1 or num_trades > 10:
        await interaction.response.send_message("❌ Number of trades must be between 1 and 10.")
        return
    
    # Defer response as trading may take time
    await interaction.response.defer()
    
    embed = discord.Embed(
        title="📈 Starting Dry Trading",
        description=f"Executing **{num_trades}** paper trade(s)...",
        color=0x0099ff
    )
    embed.add_field(name="Mode", value="🧪 Paper Trading (No Real Money)", inline=True)
    embed.add_field(name="Budget", value=f"💰 ${dry_trade_budget:.2f} USDT", inline=True)
    
    await interaction.followup.send(embed=embed)
    
    # Execute trades
    total_pnl = 0
    successful_trades = 0
    
    for i in range(num_trades):
        try:
            result = run_single_trade()
            if result:
                total_pnl += result.get('pnl_amount', 0)
                successful_trades += 1
                
                # Send update for each trade
                trade_embed = discord.Embed(
                    title=f"📊 Trade {i+1}/{num_trades} Complete",
                    color=0x00ff00 if result.get('pnl_amount', 0) > 0 else 0xff0000
                )
                trade_embed.add_field(name="Coin", value=result.get('coin', 'Unknown'), inline=True)
                trade_embed.add_field(name="PnL", value=f"${result.get('pnl_amount', 0):.2f}", inline=True)
                trade_embed.add_field(name="PnL %", value=f"{result.get('pnl_percent', 0):.2f}%", inline=True)
                
                await interaction.followup.send(embed=trade_embed)
        except Exception as e:
            logger.error(f"Trade {i+1} failed: {e}")
    
    # Send summary
    summary_embed = discord.Embed(
        title="📈 Dry Trading Complete",
        description=f"Completed **{successful_trades}/{num_trades}** trades",
        color=0x00ff00 if total_pnl > 0 else 0xff0000
    )
    summary_embed.add_field(name="Total PnL", value=f"${total_pnl:.2f}", inline=True)
    summary_embed.add_field(name="Success Rate", value=f"{(successful_trades/num_trades)*100:.1f}%", inline=True)
    
    await interaction.followup.send(embed=summary_embed)

@bot.tree.command(name="train_model", description="Train a new trading model")
@app_commands.describe(model_type="Type of model to train (random_forest, xgboost)")
async def train_model(interaction: discord.Interaction, model_type: str = "random_forest"):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized.")
        return
    
    if not MODEL_TRAINING_AVAILABLE:
        await interaction.response.send_message("❌ **Model training not available** in this deployment.\nThis may be a lightweight version.")
        return
    
    # Defer response as training may take time
    await interaction.response.defer()
    
    embed = discord.Embed(
        title="🤖 Starting Model Training",
        description=f"Training **{model_type}** model...",
        color=0x9900ff
    )
    embed.add_field(name="Model Type", value=model_type, inline=True)
    embed.add_field(name="Status", value="⏳ Training in progress", inline=True)
    embed.add_field(name="Note", value="This may take several minutes", inline=False)
    
    await interaction.followup.send(embed=embed)
    
    try:
        # Start training in background
        if model_type.lower() in ["random_forest", "rf"]:
            await asyncio.get_event_loop().run_in_executor(None, train_rf_model)
        else:
            await interaction.followup.send("❌ Unsupported model type. Use 'random_forest' or 'rf'")
            return
        
        # Training complete
        completion_embed = discord.Embed(
            title="✅ Model Training Complete",
            description=f"**{model_type}** model has been trained successfully!",
            color=0x00ff00
        )
        completion_embed.add_field(name="Next Steps", value="Use `/status` to check model performance", inline=False)
        
        await interaction.followup.send(embed=completion_embed)
        
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Training Failed",
            description=f"Model training encountered an error: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=error_embed)

@bot.tree.command(name="balance", description="Check current USDT balance")
async def balance(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized.")
        return
    
    if not TRADING_AVAILABLE:
        await interaction.response.send_message("❌ **Trading system not available** in this deployment.")
        return
    
    try:
        current_balance = get_usdt_balance()
        
        embed = discord.Embed(
            title="💰 Account Balance",
            color=0x00ff00
        )
        embed.add_field(name="USDT Balance", value=f"${current_balance:.2f}", inline=True)
        embed.add_field(name="Status", value="✅ Available for trading", inline=True)
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        await interaction.response.send_message(f"❌ **Error checking balance**: {str(e)}")

@bot.tree.command(name="trading_stats", description="View trading performance statistics")
async def trading_stats(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized.")
        return
    
    if not TRADING_AVAILABLE or not TRADING_STATS_AVAILABLE:
        await interaction.response.send_message("❌ **Trading system not available** in this deployment.")
        return
    
    try:
        stats_manager = get_stats_manager()
        stats = stats_manager.get_summary_stats()
        
        embed = discord.Embed(
            title="📊 Trading Performance",
            color=0x0099ff
        )
        
        embed.add_field(name="Total Trades", value=stats.get('total_trades', 0), inline=True)
        embed.add_field(name="Win Rate", value=f"{stats.get('win_rate', 0):.1f}%", inline=True)
        embed.add_field(name="Total PnL", value=f"${stats.get('total_pnl', 0):.2f}", inline=True)
        embed.add_field(name="Best Trade", value=f"${stats.get('best_trade', 0):.2f}", inline=True)
        embed.add_field(name="Worst Trade", value=f"${stats.get('worst_trade', 0):.2f}", inline=True)
        embed.add_field(name="Avg Trade", value=f"${stats.get('avg_trade', 0):.2f}", inline=True)
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        await interaction.response.send_message(f"❌ **Error fetching stats**: {str(e)}")

@bot.tree.command(name="help", description="Show all available commands")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🤖 Enhanced Trading Bot Commands",
        description="Complete command reference for the Railway-deployed trading system",
        color=0x0099ff
    )
    
    # Basic Commands
    embed.add_field(
        name="🔧 Basic Commands",
        value="`/ping` - Check bot responsiveness\n"
              "`/status` - View comprehensive system status\n"
              "`/help` - Show this help message",
        inline=False
    )
    
    # Data & Scraping Commands
    embed.add_field(
        name="📊 Data Collection",
        value="`/start_scraper` - Start cryptocurrency data scraper\n"
              "`/stop_scraper` - Stop the data scraper",
        inline=False
    )
    
    # Model Training Commands
    embed.add_field(
        name="🤖 Machine Learning",
        value="`/train_model [type]` - Train a new trading model\n"
              "  └ Supported types: random_forest, xgboost",
        inline=False
    )
    
    # Trading Commands
    embed.add_field(
        name="💰 Trading",
        value="`/start_dry_trade [num]` - Start paper trading (1-10 trades)\n"
              "`/balance` - Check current USDT balance\n"
              "`/trading_stats` - View performance statistics",
        inline=False
    )
    
    # System Info
    embed.add_field(
        name="ℹ️ System Information",
        value="**Deployment**: Railway Cloud\n"
              "**Mode**: Enhanced Lightweight\n"
              "**Authorization**: Required for all commands",
        inline=False
    )
    
    # Getting Started Guide
    embed.add_field(
        name="🚀 Getting Started",
        value="1️⃣ Use `/start_scraper` to collect market data\n"
              "2️⃣ Use `/train_model` to create trading models\n"
              "3️⃣ Use `/start_dry_trade` to test trading strategies\n"
              "4️⃣ Monitor with `/status` and `/trading_stats`",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="deploy_test", description="Test Railway deployment functionality")
async def deploy_test(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("🛑 You are not authorized.")
        return
    
    embed = discord.Embed(
        title="🚀 Railway Deployment Test",
        description="Testing enhanced bot deployment functionality",
        color=0x00ff00
    )
    
    # Test environment variables
    env_vars = []
    if os.getenv("DISCORD_BOT_TOKEN"):
        env_vars.append("✅ DISCORD_BOT_TOKEN")
    if os.getenv("DISCORD_USER_ID"):
        env_vars.append("✅ DISCORD_USER_ID")
    
    embed.add_field(name="Environment Variables", value="\n".join(env_vars) or "❌ No vars set", inline=False)
    
    # System capabilities
    capabilities = []
    if TRADING_AVAILABLE:
        capabilities.append("✅ Trading System")
    else:
        capabilities.append("❌ Trading System")
    
    if SCRAPER_AVAILABLE:
        capabilities.append("✅ Data Scraper")
    else:
        capabilities.append("❌ Data Scraper")
    
    if MODEL_TRAINING_AVAILABLE:
        capabilities.append("✅ Model Training")
    else:
        capabilities.append("❌ Model Training")
    
    embed.add_field(name="System Capabilities", value="\n".join(capabilities), inline=False)
    embed.add_field(name="Python Version", value=f"🐍 {sys.version.split()[0]}", inline=True)
    embed.add_field(name="Discord.py Version", value=f"📦 {discord.__version__}", inline=True)
    embed.add_field(name="Port", value=f"🌐 {PORT}", inline=True)
    
    await interaction.response.send_message(embed=embed)

async def health_check(request):
    """Health check endpoint for Railway"""
    status = {
        "status": "healthy",
        "service": "lightweight-discord-bot",
        "health_server": "running",
        "discord_bot": "connected" if bot.is_ready() else "disconnected",
        "timestamp": asyncio.get_event_loop().time()
    }
    
    # Return JSON response for better debugging
    return web.json_response(status, status=200)

async def start_health_server():
    """Start health check server on the port specified by Railway"""
    try:
        app = web.Application()
        app.router.add_get('/health', health_check)
        app.router.add_get('/', health_check)  # Also respond to root path
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        # Bind to all interfaces (0.0.0.0) and use Railway's PORT
        site = web.TCPSite(runner, '0.0.0.0', PORT)
        await site.start()
        
        logger.info(f"🏥 Health check server started on 0.0.0.0:{PORT}")
        logger.info(f"🌐 Health endpoints: http://0.0.0.0:{PORT}/health and http://0.0.0.0:{PORT}/")
        logger.info(f"🚀 Railway health check endpoint ready at /health")
        
        return runner
        
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"❌ Port {PORT} is already in use")
            raise
        else:
            logger.error(f"❌ Failed to start health server on port {PORT}: {e}")
            raise
    except Exception as e:
        logger.error(f"❌ Unexpected error starting health server: {e}")
        raise

async def main_async():
    """Async main function to run both services"""
    health_runner = None
    
    try:
        logger.info("🔄 Initializing health server for Railway...")
        
        # Start health check server FIRST - this is critical for Railway
        health_runner = await start_health_server()
        logger.info("✅ Health server ready for Railway health checks")
        
        # Give health server extra time to fully initialize for Railway
        logger.info("⏳ Waiting for health server to stabilize...")
        await asyncio.sleep(2)
        
        # Test health endpoint internally
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{PORT}/health') as resp:
                    if resp.status == 200:
                        logger.info("✅ Internal health check passed")
                    else:
                        logger.warning(f"⚠️ Internal health check returned {resp.status}")
        except Exception as e:
            logger.warning(f"⚠️ Internal health check failed: {e}")
        
        # Only start Discord bot if token is present and valid
        if TOKEN and TOKEN != "dummy" and TOKEN != "":
            logger.info("🤖 Starting Discord bot...")
            await bot.start(TOKEN)
        else:
            logger.warning("⚠️ No valid Discord token - running health server only")
            logger.info("🔄 Entering health-only mode for Railway...")
            
            # Keep the health server running indefinitely
            while True:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                logger.debug(f"💓 Health server heartbeat on port {PORT}")
        
    except discord.LoginFailure as e:
        logger.error(f"❌ Discord login failed: {e}")
        logger.info("⚡ Keeping health server running for Railway...")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.error(f"📄 Traceback: {traceback.format_exc()}")
        logger.info("⚡ Keeping health server running for Railway...")
    
    # Keep the health server running indefinitely for Railway
    logger.info("🔄 Entering maintenance mode - health checks will continue...")
    try:
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            logger.debug(f"💓 Health server heartbeat on port {PORT}")
    except KeyboardInterrupt:
        logger.info("👋 Received shutdown signal")
    finally:
        if health_runner:
            logger.info("🛑 Shutting down health server...")
            await health_runner.cleanup()

def main():
    """Main entry point for the enhanced lightweight Discord bot."""
    logger.info("🚀 Starting Enhanced Discord Bot for Railway...")
    logger.info(f"🌐 Health server will start on port {PORT}")
    
    # Validate environment variables
    if not TOKEN:
        logger.error("❌ Discord bot token not found! Set DISCORD_BOT_TOKEN environment variable.")
        logger.info("⚡ Starting health server anyway for Railway health checks...")
    else:
        logger.info("✅ Discord bot token found")
    
    if AUTHORIZED_USER == 0:
        logger.warning("⚠️ DISCORD_USER_ID not set - commands will be unrestricted")
    else:
        logger.info(f"✅ Authorized user ID: {AUTHORIZED_USER}")
    
    # Log system capabilities
    logger.info("🔧 System Capabilities:")
    logger.info(f"  📈 Trading: {'✅ Available' if TRADING_AVAILABLE else '❌ Disabled'}")
    logger.info(f"  📊 Data Scraper: {'✅ Available' if SCRAPER_AVAILABLE else '❌ Disabled'}")
    logger.info(f"  🤖 Model Training: {'✅ Available' if MODEL_TRAINING_AVAILABLE else '❌ Disabled'}")
    
    try:
        # Set up event loop for better error handling
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async main function
        loop.run_until_complete(main_async())
        
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(f"📄 Full traceback: {traceback.format_exc()}")
        logger.info("🚨 Bot will exit - Railway will restart the container")
        # Don't re-raise the exception - let the process exit cleanly
        sys.exit(1)
    finally:
        # Clean up the event loop
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except:
            pass

if __name__ == "__main__":
    main()

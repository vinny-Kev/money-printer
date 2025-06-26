import os
import sys
import discord
from discord.ext import commands
from dotenv import load_dotenv
from datetime import datetime

# Fix imports for standalone execution
try:
    from .trade_runner import run_single_trade, get_usdt_balance
    from ..trading_stats import get_stats_manager
    from ..model_training.trainer_diagnostics import get_trainer_diagnostics
    from ..auto_culling import get_auto_culler
    from src.railway_watchdog import get_railway_watchdog
    from src.drive_manager import get_drive_manager
except ImportError:
    # Add parent directory to path for standalone execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.trading_bot.trade_runner import run_single_trade, get_usdt_balance
    from src.trading_stats import get_stats_manager
    from src.model_training.trainer_diagnostics import get_trainer_diagnostics
    from src.auto_culling import get_auto_culler
    from src.railway_watchdog import get_railway_watchdog
    from src.drive_manager import get_drive_manager

load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
AUTHORIZED_USER = int(os.getenv("DISCORD_USER_ID", "0"))  # Your Discord user ID

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="/", intents=intents)

@bot.event
async def on_ready():
    print(f"⚡ Logged in as {bot.user}")
      # Display dashboard on startup
    try:
        stats_mgr = get_stats_manager()
        from .trade_runner import dry_trade_budget
        balance = dry_trade_budget  # Use the dry trading budget instead of real balance
        dashboard = stats_mgr.format_dashboard_display(balance)
        print("\n" + dashboard)
    except Exception as e:
        print(f"⚠️ Could not load dashboard on startup: {e}")
        print("💡 Dashboard will be available once trading begins.")

@bot.command(name="start_dry_trade")
async def start_dry_trade(ctx, count: int = 1):
    """
    Start dry trading for the specified number of trades.
    Defaults to 1 trade if no count is provided or invalid.
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to command the markets.")
        return

    if count <= 0:
        count = 1  # Default to 1 trade if invalid input

    await ctx.send(f"📈 Starting {count} dry trade{'s' if count > 1 else ''}, my lord...")

    for i in range(count):
        await ctx.send(f"🎯 Executing Trade {i + 1}/{count}...")

        # Run your core trader logic (this should BLOCK until the trade completes)
        receipt = run_single_trade()

        # Send results
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

        await ctx.send(embed=embed)

    await ctx.send(f"✅ All {count} trades complete.")

@bot.command(name="start_live_trade")
async def start_live_trade(ctx):
    """
    Start live trading.
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to command the markets.")
        return

    balance = get_usdt_balance()
    if balance is None or balance < 10:  # Assuming $10 is the minimum balance
        await ctx.send(
            f"⚠️ My lord, we're out of money for live trading. Consider switching to dry trading instead."
        )
        return

    await ctx.send("📈 Starting live trading, my lord...")    # Run your core trader logic (this should BLOCK until the trade completes)
    receipt = run_single_trade()

    # Send results
    embed = discord.Embed(
        title="📜 Trade Receipt",
        description="My lord, I have finished my work. This is my harvest.",
        color=0x00ff00
    )
    embed.add_field(name="Coin", value=receipt["coin"], inline=True)
    embed.add_field(name="Side", value=receipt["side"], inline=True)
    embed.add_field(name="Buy Price", value=f"${receipt['buy_price']}", inline=True)
    embed.add_field(name="Sell Price", value=f"${receipt['sell_price']}", inline=True)
    embed.add_field(name="PnL", value=f"{receipt['pnl_percent']}% | ${receipt['pnl_amount']}", inline=False)
    embed.set_footer(text="🪙 Taxable Income: Log this.")

    await ctx.send(embed=embed)

@bot.command(name="dashboard")
async def dashboard(ctx):
    """
    Display the trading dashboard.
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to view the dashboard.")
        return

    try:
        stats_mgr = get_stats_manager()
        balance = get_usdt_balance()
        dashboard = stats_mgr.format_dashboard_display(balance)

        await ctx.send("📊 Trading Dashboard:")
        await ctx.send(f"```\n{dashboard}\n```")
    except Exception as e:
        await ctx.send(f"🚨 Could not retrieve dashboard: {e}")

@bot.command(name="leaderboard")
async def leaderboard(ctx):
    """
    Display the model performance leaderboard.
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to view the leaderboard.")
        return

    try:
        stats_mgr = get_stats_manager()
        leaderboard_data = stats_mgr.get_model_leaderboard()

        if not leaderboard_data:
            await ctx.send("📉 No model performance data available.")
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

        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"🚨 Could not retrieve leaderboard: {e}")

@bot.command(name="status")
async def status(ctx):
    """
    Display comprehensive trading status.
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to view status.")
        return

    try:
        stats_mgr = get_stats_manager()
        balance = get_usdt_balance()
        dashboard = stats_mgr.format_dashboard_display(balance)

        await ctx.send("📊 **TRADING STATUS**")
        await ctx.send(f"```\n{dashboard}\n```")
    except Exception as e:
        await ctx.send(f"🚨 Could not retrieve status: {e}")

@bot.command(name="metrics")
async def metrics(ctx, model_name: str = None):
    """
    Get detailed metrics for a specific model.
    Usage: /metrics [model_name]
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to view metrics.")
        return

    try:
        stats_mgr = get_stats_manager()
        
        if not model_name:
            # Show available models
            models = list(stats_mgr.models_performance.keys())
            if models:
                await ctx.send(f"📊 Available models: {', '.join(models)}\nUsage: `/metrics [model_name]`")
            else:
                await ctx.send("📊 No models available.")
            return

        diagnostics = stats_mgr.get_model_diagnostics(model_name.lower())
        
        if not diagnostics:
            await ctx.send(f"❌ Model '{model_name}' not found.")
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
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"🚨 Could not retrieve metrics: {e}")

@bot.command(name="retrain")
async def retrain(ctx, target: str = "weak"):
    """
    Initiate model retraining.
    Usage: /retrain [weak|all|model_name]
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to initiate retraining.")
        return

    try:
        stats_mgr = get_stats_manager()
        if target.lower() == "weak":
            # Retrain only underperforming models using auto-culler
            culler = get_auto_culler()
            weak_models = stats_mgr.get_underperforming_models()
            
            if not weak_models:
                await ctx.send("✅ No underperforming models found.")
                return
                
            await ctx.send(f"🔄 Initiating retraining for flagged models: {', '.join(weak_models)}")
            
            # Use auto-culler to attempt retraining
            success_count = 0
            for model in weak_models:
                if culler.attempt_retrain(model):
                    success_count += 1
                    
            await ctx.send(f"✅ Successfully retrained {success_count}/{len(weak_models)} models.")
            
        elif target.lower() == "all":
            await ctx.send("🔄 Initiating full model retraining...")
            
            # Use auto-culler to retrain all models
            culler = get_auto_culler()
            all_models = list(stats_mgr.models_performance.keys())
            success_count = 0
            
            for model in all_models:
                if culler.attempt_retrain(model):
                    success_count += 1
                    
            await ctx.send(f"✅ Retrained {success_count}/{len(all_models)} models.")
            
        else:
            # Retrain specific model
            if target.lower() in stats_mgr.models_performance:
                await ctx.send(f"🔄 Initiating retraining for {target.upper()}...")
                
                culler = get_auto_culler()
                if culler.attempt_retrain(target.lower()):
                    await ctx.send(f"✅ Successfully retrained {target.upper()}.")
                else:
                    await ctx.send(f"❌ Retraining failed for {target.upper()}.")
            else:
                await ctx.send(f"❌ Model '{target}' not found.")
                
    except Exception as e:
        await ctx.send(f"🚨 Retraining failed: {e}")

@bot.command(name="balance")
async def balance(ctx):
    """
    Check current wallet balance.
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to view balance.")
        return

    try:
        balance = get_usdt_balance()
        embed = discord.Embed(
            title="💰 Wallet Balance",
            description=f"Current USDT Balance: **${balance:.2f}**",
            color=0x00ff00
        )
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"🚨 Could not retrieve balance: {e}")

@bot.command(name="culling")
async def culling(ctx, action: str = "status"):
    """
    Manage auto-culling system.
    Usage: /culling [status|check|enable|disable|unpause <model>]
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to manage auto-culling.")
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
            embed.add_field(name="Paused Models", value=str(status['paused_models']), inline=True)
            
            if status['paused_details']:
                paused_info = []
                for model, info in status['paused_details'].items():
                    resume_time = datetime.fromisoformat(info['pause_until']).strftime('%H:%M')
                    paused_info.append(f"• {model}: {info['reason']} (until {resume_time})")
                
                embed.add_field(
                    name="Paused Models Details", 
                    value='\n'.join(paused_info[:5]),  # Show max 5
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        elif action.lower() == "check":
            await ctx.send("🔍 Running auto-culling performance check...")
            culler.run_culling_check()
            await ctx.send("✅ Auto-culling check completed.")
            
        elif action.lower() == "enable":
            culler.update_config({"enabled": True})
            await ctx.send("✅ Auto-culling system enabled.")
            
        elif action.lower() == "disable":
            culler.update_config({"enabled": False})
            await ctx.send("⚠️ Auto-culling system disabled.")
            
        else:
            await ctx.send(f"❌ Unknown action: {action}. Use: status, check, enable, disable")
                
    except Exception as e:
        await ctx.send(f"🚨 Auto-culling command failed: {e}")

@bot.command(name="unpause")
async def unpause(ctx, model_name: str):
    """
    Manually unpause a model.
    Usage: /unpause <model_name>
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to unpause models.")
        return

    try:
        culler = get_auto_culler()
        
        if culler.unpause_model(model_name.lower()):
            await ctx.send(f"🔓 Model {model_name.upper()} has been unpaused.")
        else:
            await ctx.send(f"❌ Model {model_name.upper()} was not paused.")
                
    except Exception as e:
        await ctx.send(f"🚨 Unpause failed: {e}")

@bot.command(name="stop_trading")
async def stop_trading(ctx):
    """
    Emergency stop all trading operations.
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to stop trading.")
        return

    try:
        # Create emergency stop flag
        with open("TRADING_DISABLED.flag", "w") as f:
            f.write(f"Emergency stop initiated by Discord user at {datetime.utcnow().isoformat()}")
            
        await ctx.send("🚨 **EMERGENCY STOP ACTIVATED**\nAll trading operations have been halted.")
        
    except Exception as e:
        await ctx.send(f"🚨 Emergency stop failed: {e}")

@bot.command(name="usage_status")
async def usage_status(ctx):
    """
    Check Railway usage status and remaining hours
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to check usage status.")
        return

    try:
        watchdog = get_railway_watchdog()
        status = watchdog.get_usage_status()
        
        if "error" in status:
            await ctx.send(f"❌ Failed to get usage data: {status['error']}")
            return
        
        # Create status embed
        if status['status'] == 'critical':
            color = 0xff0000  # Red
            emoji = "🚨"
        elif status['status'] == 'warning':
            color = 0xffa500  # Orange
            emoji = "⚠️"
        else:
            color = 0x00ff00  # Green
            emoji = "✅"
        
        embed = discord.Embed(
            title=f"{emoji} Railway Usage Status",
            color=color
        )
        
        embed.add_field(
            name="📊 Current Usage",
            value=f"{status['current_hours']:.2f} / {status['limit_hours']} hours",
            inline=True
        )
        
        embed.add_field(
            name="📈 Usage Percentage",
            value=f"{status['usage_percentage']:.1f}%",
            inline=True
        )
        
        embed.add_field(
            name="⏰ Remaining",
            value=f"{status['remaining_hours']:.2f} hours",
            inline=True
        )
        
        embed.add_field(
            name="💰 Estimated Cost",
            value=f"${status['estimated_cost']:.2f}",
            inline=True
        )
        
        embed.add_field(
            name="📅 Billing Cycle",
            value=f"Until {status['billing_cycle_end'][:10]}",
            inline=True
        )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"🚨 Could not retrieve usage status: {e}")

@bot.command(name="drive_status")
async def drive_status(ctx):
    """
    Check Google Drive sync status
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to check Drive status.")
        return

    try:
        manager = get_drive_manager()
        status = manager.get_status()
        
        # Create status embed
        if status['enabled'] and status['authenticated']:
            color = 0x00ff00  # Green
            emoji = "✅"
            status_text = "Active"
        elif status['enabled'] and not status['authenticated']:
            color = 0xffa500  # Orange
            emoji = "⚠️"
            status_text = "Authentication Required"
        else:
            color = 0x808080  # Gray
            emoji = "⏸️"
            status_text = "Disabled"
        
        embed = discord.Embed(
            title=f"{emoji} Google Drive Sync Status",
            description=f"Status: **{status_text}**",
            color=color
        )
        
        embed.add_field(
            name="🔐 Authenticated",
            value="Yes" if status['authenticated'] else "No",
            inline=True
        )
        
        embed.add_field(
            name="📁 Folder ID",
            value=status['folder_id'][:20] + "..." if status['folder_id'] else "Not set",
            inline=True
        )
        
        embed.add_field(
            name="📊 Cached Files",
            value=str(status['cached_files']),
            inline=True
        )
        
        if status['last_sync']:
            embed.add_field(
                name="⏰ Last Sync",
                value=status['last_sync'][:19].replace('T', ' '),
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"🚨 Could not retrieve Drive status: {e}")

@bot.command(name="drive_sync")
async def drive_sync(ctx):
    """
    Manually trigger Google Drive sync
    """
    if ctx.author.id != AUTHORIZED_USER:
        await ctx.send("🛑 You are not authorized to trigger sync.")
        return

    try:
        manager = get_drive_manager()
        
        if not manager.sync_enabled:
            await ctx.send("⏸️ Google Drive sync is disabled.")
            return
            
        if not manager.authenticated:
            await ctx.send("❌ Google Drive not authenticated. Please set up credentials.")
            return
        
        await ctx.send("🔄 Starting manual sync...")
        
        results = manager.sync_trading_data()
        
        if "error" in results:
            await ctx.send(f"❌ Sync failed: {results['error']}")
            return
        
        total_synced = sum([
            results.get('models', 0),
            results.get('trades', 0),
            results.get('market_data', 0),
            results.get('diagnostics', 0),
            results.get('stats', 0),
            results.get('logs', 0)        ])
        
        embed = discord.Embed(
            title="📁 Sync Complete",
            description=f"Successfully synced {total_synced} files",
            color=0x00ff00
        )
        embed.add_field(name="🤖 Models", value=str(results.get('models', 0)), inline=True)
        embed.add_field(name="💹 Trades", value=str(results.get('trades', 0)), inline=True)
        embed.add_field(name="📊 Diagnostics", value=str(results.get('diagnostics', 0)), inline=True)
        embed.add_field(name="📈 Market Data", value=str(results.get('market_data', 0)), inline=True)
        embed.add_field(name="📋 Stats", value=str(results.get('stats', 0)), inline=True)
        embed.add_field(name="📄 Logs", value=str(results.get('logs', 0)), inline=True)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"🚨 Sync failed: {e}")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("❌ Command not found. Use `/help` to see available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("⚠️ Missing required argument. Please check the command usage.")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("⚠️ Invalid argument. Please provide the correct input.")
    else:
        await ctx.send("🚨 An unexpected error occurred. Please try again later.")
        print(f"Discord bot error: {error}")  # Log error for debugging


bot.run(TOKEN)

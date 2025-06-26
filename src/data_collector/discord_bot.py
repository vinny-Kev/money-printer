import os
import time
import logging
import threading
import random  # Import random for selecting quotes
from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.ext import commands

# Import data scraper functions
try:
    from .data_scraper import main as start_scraper, handle_sigint
except ImportError:
    # If running from outside the package, try absolute import
    try:
        from src.data_collector.data_scraper import main as start_scraper, handle_sigint
    except ImportError:
        # Last resort - add path and import
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        sys.path.append(os.path.dirname(__file__))
        from data_scraper import main as start_scraper, handle_sigint

# Load environment variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

if not DISCORD_BOT_TOKEN:
    raise ValueError("Missing DISCORD_BOT_TOKEN in the .env file.")
if not DISCORD_CHANNEL_ID:
    raise ValueError("Missing DISCORD_CHANNEL_ID in the .env file.")
else:
    DISCORD_CHANNEL_ID = int(DISCORD_CHANNEL_ID)

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DiscordScraperBot")

# Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)
tree = bot.tree  # Access the app_commands tree

# Global variables to control the scraper
scraper_thread = None
scraper_running = False

# Peasant quotes for starting the scraper
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

# Stop quotes for stopping the scraper
STOP_QUOTES = [
    "The bytes rest, my lord.",
    "The peasants return to the shadows.",
    "As you command, the scanner slumbers again.",
    "The winds grow silent... the scraper rests.",
]

def stop_scraper():
    """
    Stop the scraper gracefully.
    """
    global scraper_running
    if scraper_running:
        logger.info("Stopping the scraper...")
        scraper_running = False
        handle_sigint(None, None)  # Trigger graceful shutdown in the scraper
        logger.info("Scraper stopped.")
    else:
        logger.info("Scraper is not running.")

def scraper_runner(duration=None):
    """
    Run the scraper for a specified duration or indefinitely.
    :param duration: Duration in hours to run the scraper (None for indefinite).
    """
    global scraper_running
    scraper_running = True
    try:
        if duration:
            logger.info(f"Running scraper for {duration} hours...")
            start_scraper()  # Call the scraper once; it handles its own infinite loop
        else:
            start_scraper()  # Run indefinitely
    except Exception as e:
        logger.error(f"Scraper error: {e}")
    finally:
        scraper_running = False
        logger.info("Scraper thread finished.")

@tree.command(name="start_scraper", description="Start the scraper for a specified duration (in hours).")
async def start_scraper_command(interaction: discord.Interaction, hours: int = None):
    """
    Slash command to start the scraper.
    :param interaction: The interaction object.
    :param hours: Optional duration in hours to run the scraper.
    """
    global scraper_thread

    if scraper_running:
        await interaction.response.send_message("📡 Scraper is already running, my lord.")
        return

    # Select a random quote
    battle_quote = random.choice(PEASANT_QUOTES)

    # Start the scraper in a separate thread
    scraper_thread = threading.Thread(target=scraper_runner, args=(hours,))
    scraper_thread.start()
    await interaction.response.send_message(f"📜 {battle_quote}\n\nScraper started for {'indefinite' if hours is None else f'{hours} hours'}.")

@tree.command(name="stop_scraper", description="Stop the scraper.")
async def stop_scraper_command(interaction: discord.Interaction):
    """
    Slash command to stop the scraper.
    :param interaction: The interaction object.
    """
    global scraper_thread
    if not scraper_running:
        await interaction.response.send_message("⚠️ Scraper is not running.")
        return

    # Select a random stop quote
    stop_quote = random.choice(STOP_QUOTES)

    stop_scraper()
    scraper_thread.join(timeout=5)  # Wait for the thread to finish
    scraper_thread = None  # Reset the thread for future runs
    await interaction.response.send_message(f"🛑 {stop_quote}")

@tree.command(name="status", description="Check the status of the scraper.")
async def status_command(interaction: discord.Interaction):
    """
    Slash command to check the status of the scraper.
    :param interaction: The interaction object.
    """
    if scraper_running:
        await interaction.response.send_message("📈 Scraper is currently running.")
    else:
        await interaction.response.send_message("💤 Scraper is idle.")

@bot.event
async def on_ready():
    """
    Event triggered when the bot is ready.
    """
    logger.info(f"Logged in as {bot.user.name}")
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        await channel.send("📁 **Local Scraper Bot Online** – Ready to deploy local storage scanners.")
    else:
        logger.warning("❌ Could not find the specified channel. Check DISCORD_CHANNEL_ID and bot permissions.")

    # Sync slash commands with Discord
    try:
        await tree.sync()
        logger.info("✅ Slash commands synced with Discord.")
    except Exception as e:
        logger.error(f"❌ Failed to sync slash commands: {e}")

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
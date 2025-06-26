#!/usr/bin/env python3
"""
Trading Bot Training Discord Bot - Remote ML Pipeline Control

This Discord bot provides remote control over the machine learning training pipeline,
allowing you to train models, check training status, and manage the ML workflow.
"""

import os
import sys
import asyncio
import logging
import threading
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.ext import commands

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.model_training.incremental_trainer import IncrementalTrainer
from src.model_training.random_forest_trainer import main as train_random_forest
from src.model_variants.xgboost_trainer import main as train_xgboost
from src.discord_notifications import send_trainer_notification

# Load environment variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID")

if not DISCORD_BOT_TOKEN:
    raise ValueError("Missing DISCORD_BOT_TOKEN in the .env file.")
if not DISCORD_CHANNEL_ID:
    raise ValueError("Missing DISCORD_CHANNEL_ID in the .env file.")
else:
    DISCORD_CHANNEL_ID = int(DISCORD_CHANNEL_ID)

if DISCORD_USER_ID:
    DISCORD_USER_ID = int(DISCORD_USER_ID)

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainingBot")

# Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# Training status tracking
training_status = {
    "is_training": False,
    "current_model": None,
    "start_time": None,
    "last_update": None
}

def is_authorized(user_id):
    """Check if user is authorized to use training commands"""
    if DISCORD_USER_ID is None:
        return True  # No restriction if DISCORD_USER_ID not set
    return user_id == DISCORD_USER_ID

@bot.event
async def on_ready():
    """Event triggered when the bot is ready."""
    logger.info(f"Logged in as {bot.user.name}")
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")
    
    # Send startup message
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        embed = discord.Embed(
            title="🤖 Training Bot Online",
            description="Ready to control ML training pipeline",
            color=0x00ff00
        )
        embed.add_field(name="Available Commands", 
                        value="/train_rf - Train Random Forest\n/train_xgb - Train XGBoost\n/incremental - Incremental training\n/status - Check training status\n/stop_training - Stop current training",
                        inline=False)
        await channel.send(embed=embed)
    else:
        logger.warning("Could not find the specified channel.")

def run_training_task(model_type, interaction_user_id, channel_id):
    """Run training task in background thread"""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["current_model"] = model_type
        training_status["start_time"] = datetime.now()
        training_status["last_update"] = datetime.now()
        
        logger.info(f"Starting {model_type} training...")
        
        if model_type == "random_forest":
            train_random_forest()
        elif model_type == "xgboost":
            train_xgboost()
        elif model_type == "incremental":
            trainer = IncrementalTrainer()
            trainer.retrain_model()
        
        # Send completion notification
        send_trainer_notification(f"✅ **Training Complete**: {model_type} model training finished successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        send_trainer_notification(f"❌ **Training Failed**: {model_type} - Error: {str(e)}")
    finally:
        training_status["is_training"] = False
        training_status["current_model"] = None
        training_status["start_time"] = None

@bot.tree.command(name="train_rf", description="Train Random Forest model")
async def train_random_forest_command(interaction: discord.Interaction):
    """Train Random Forest model"""
    if not is_authorized(interaction.user.id):
        await interaction.response.send_message("❌ You are not authorized to use this command.", ephemeral=True)
        return
    
    if training_status["is_training"]:
        await interaction.response.send_message(f"⚠️ Training already in progress: {training_status['current_model']}", ephemeral=True)
        return
    
    await interaction.response.send_message("🚀 Starting Random Forest training...")
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_task, args=("random_forest", interaction.user.id, interaction.channel.id))
    thread.daemon = True
    thread.start()

@bot.tree.command(name="train_xgb", description="Train XGBoost model")
async def train_xgboost_command(interaction: discord.Interaction):
    """Train XGBoost model"""
    if not is_authorized(interaction.user.id):
        await interaction.response.send_message("❌ You are not authorized to use this command.", ephemeral=True)
        return
    
    if training_status["is_training"]:
        await interaction.response.send_message(f"⚠️ Training already in progress: {training_status['current_model']}", ephemeral=True)
        return
    
    await interaction.response.send_message("🚀 Starting XGBoost training...")
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_task, args=("xgboost", interaction.user.id, interaction.channel.id))
    thread.daemon = True
    thread.start()

@bot.tree.command(name="incremental", description="Run incremental training with recent trade data")
async def incremental_training_command(interaction: discord.Interaction):
    """Run incremental training"""
    if not is_authorized(interaction.user.id):
        await interaction.response.send_message("❌ You are not authorized to use this command.", ephemeral=True)
        return
    
    if training_status["is_training"]:
        await interaction.response.send_message(f"⚠️ Training already in progress: {training_status['current_model']}", ephemeral=True)
        return
    
    await interaction.response.send_message("🔄 Starting incremental training...")
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_task, args=("incremental", interaction.user.id, interaction.channel.id))
    thread.daemon = True
    thread.start()

@bot.tree.command(name="status", description="Check training status")
async def training_status_command(interaction: discord.Interaction):
    """Check training status"""
    embed = discord.Embed(title="🤖 Training Status", color=0x0099ff)
    
    if training_status["is_training"]:
        duration = datetime.now() - training_status["start_time"]
        embed.add_field(name="Status", value="🟡 Training in Progress", inline=True)
        embed.add_field(name="Current Model", value=training_status["current_model"], inline=True)
        embed.add_field(name="Duration", value=f"{duration.total_seconds():.0f}s", inline=True)
    else:
        embed.add_field(name="Status", value="🟢 Idle", inline=True)
        embed.add_field(name="Current Model", value="None", inline=True)
        
        # Check for recent models
        models_dir = "data/models"
        if os.path.exists(models_dir):
            recent_models = []
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(('.pkl', '.joblib')):
                        filepath = os.path.join(root, file)
                        mtime = os.path.getmtime(filepath)
                        recent_models.append((file, datetime.fromtimestamp(mtime)))
            
            if recent_models:
                recent_models.sort(key=lambda x: x[1], reverse=True)
                latest = recent_models[0]
                embed.add_field(name="Latest Model", value=f"{latest[0]} ({latest[1].strftime('%Y-%m-%d %H:%M')})", inline=False)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="model_info", description="Get information about available models")
async def model_info_command(interaction: discord.Interaction):
    """Get model information"""
    embed = discord.Embed(title="📊 Model Information", color=0x9932cc)
    
    models_dir = "data/models"
    if os.path.exists(models_dir):
        model_types = {}
        
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(('.pkl', '.joblib')):
                    model_type = os.path.basename(root)
                    if model_type not in model_types:
                        model_types[model_type] = []
                    
                    filepath = os.path.join(root, file)
                    mtime = os.path.getmtime(filepath)
                    size = os.path.getsize(filepath)
                    model_types[model_type].append({
                        'name': file,
                        'modified': datetime.fromtimestamp(mtime),
                        'size': size
                    })
        
        for model_type, models in model_types.items():
            if models:
                latest = max(models, key=lambda x: x['modified'])
                embed.add_field(
                    name=f"🤖 {model_type.title()}",
                    value=f"Latest: {latest['name']}\nModified: {latest['modified'].strftime('%Y-%m-%d %H:%M')}\nSize: {latest['size']:,} bytes",
                    inline=True
                )
    else:
        embed.add_field(name="Status", value="No models directory found", inline=False)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="train_data_info", description="Get information about training data")
async def train_data_info_command(interaction: discord.Interaction):
    """Get training data information"""
    embed = discord.Embed(title="📈 Training Data Info", color=0xff6600)
    
    # Check transaction data
    transactions_dir = "data/transactions"
    if os.path.exists(transactions_dir):
        csv_files = [f for f in os.listdir(transactions_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            filepath = os.path.join(transactions_dir, csv_file)
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                
                embed.add_field(
                    name=f"📊 {csv_file}",
                    value=f"Rows: {len(df)}\nColumns: {len(df.columns)}\nModified: {datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M')}",
                    inline=True
                )
            except Exception as e:
                embed.add_field(name=f"❌ {csv_file}", value=f"Error reading: {str(e)}", inline=True)
    
    # Check scraped data
    scraped_dir = "data/scraped_data/parquet_files"
    if os.path.exists(scraped_dir):
        parquet_files = [f for f in os.listdir(scraped_dir) if f.endswith('.parquet')]
        total_size = sum(os.path.getsize(os.path.join(scraped_dir, f)) for f in parquet_files)
        
        embed.add_field(
            name="📁 Scraped Data",
            value=f"Files: {len(parquet_files)}\nTotal Size: {total_size:,} bytes",
            inline=True
        )
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="stop_training", description="Stop current training process")
async def stop_training_command(interaction: discord.Interaction):
    """Stop current training"""
    if not is_authorized(interaction.user.id):
        await interaction.response.send_message("❌ You are not authorized to use this command.", ephemeral=True)
        return
    
    if not training_status["is_training"]:
        await interaction.response.send_message("ℹ️ No training currently in progress.", ephemeral=True)
        return
    
    # Note: This is a simple flag - actual implementation would need more sophisticated process control
    training_status["is_training"] = False
    training_status["current_model"] = None
    training_status["start_time"] = None
    
    await interaction.response.send_message("🛑 Training stop signal sent.")

if __name__ == "__main__":
    logger.info("Starting Training Discord Bot...")
    bot.run(DISCORD_BOT_TOKEN)

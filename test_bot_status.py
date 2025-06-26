#!/usr/bin/env python3
"""
Quick test to check Discord bot status without starting the bot
"""

import os
import sys
import logging

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BotStatusTest")

def test_bot_status_variables():
    """Test the status variables that the bot uses"""
    
    # Import trading functionality with Railway-safe fallbacks
    try:
        from src.trading_bot.trade_runner import run_single_trade, get_usdt_balance
        TRADING_AVAILABLE = True
        logger.info("✅ Trading modules loaded successfully")
    except ImportError as e:
        logger.warning(f"⚠️ Trading modules not available: {e}")
        TRADING_AVAILABLE = False

    # Import trading stats with fallback
    try:
        from trading_stats import get_stats_manager
        TRADING_STATS_AVAILABLE = True
        logger.info("✅ Trading stats module loaded")
    except ImportError as e:
        logger.warning(f"⚠️ Trading stats not available: {e}")
        TRADING_STATS_AVAILABLE = False

    # Import scraper functionality with Railway-safe fallbacks
    try:
        from src.data_collector.data_scraper import main as start_scraper
        SCRAPER_AVAILABLE = True
        logger.info("✅ Scraper modules loaded successfully")
    except ImportError as e:
        logger.warning(f"⚠️ Scraper modules not available: {e}")
        SCRAPER_AVAILABLE = False

    # Import model training with Railway-safe fallbacks
    try:
        from src.model_training.random_forest_trainer import main as train_rf_model
        MODEL_TRAINING_AVAILABLE = True
        logger.info("✅ Model training modules loaded successfully")
    except ImportError as e:
        logger.warning(f"⚠️ Model training modules not available: {e}")
        MODEL_TRAINING_AVAILABLE = False

    logger.info("\n" + "="*50)
    logger.info("📊 BOT STATUS VARIABLES:")
    logger.info(f"  TRADING_AVAILABLE: {TRADING_AVAILABLE}")
    logger.info(f"  TRADING_STATS_AVAILABLE: {TRADING_STATS_AVAILABLE}")
    logger.info(f"  SCRAPER_AVAILABLE: {SCRAPER_AVAILABLE}")
    logger.info(f"  MODEL_TRAINING_AVAILABLE: {MODEL_TRAINING_AVAILABLE}")
    
    all_available = all([TRADING_AVAILABLE, SCRAPER_AVAILABLE, MODEL_TRAINING_AVAILABLE])
    logger.info(f"\n🎯 All core systems available: {all_available}")
    
    return {
        'trading': TRADING_AVAILABLE,
        'scraper': SCRAPER_AVAILABLE, 
        'training': MODEL_TRAINING_AVAILABLE,
        'stats': TRADING_STATS_AVAILABLE
    }

if __name__ == "__main__":
    test_bot_status_variables()

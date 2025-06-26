"""
Centralized Discord notification system for all modules.
Provides separate notification channels for different components.
"""
import os
import random
import requests
import logging
from typing import Optional
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Enumeration of different notification types with their specific webhooks."""
    GENERAL = "DISCORD_WEBHOOK"
    DATA_SCRAPER = "DISCORD_WEBHOOK_DATA_SCRAPER"
    TRAINERS = "DISCORD_WEBHOOK_TRAINERS"
    TRADERS = "DISCORD_WEBHOOK_TRADERS"

# Component-specific quote collections
SCRAPER_QUOTES = [
    "Yes My Lord! I'm on it!",
    "I will now gather information, my lord.",
    "The data shall be collected, my lord.",
    "The bytes are ready, my lord.",
    "The data is being gathered, my lord.",
    "Market surveillance initiated, my lord.",
    "Scanning the crypto markets, my lord.",
]

TRAINER_QUOTES = [
    "Training commenced, my lord!",
    "The models are learning patterns, my lord!",
    "Machine learning wisdom is being cultivated, my lord!",
    "The algorithms see all patterns, my lord!",
    "Model training in progress, my lord!",
    "The neural networks serve you well, my lord!",
]

TRADER_QUOTES = [
    "Trading operations initiated, my lord!",
    "The markets bow to your will, my lord!",
    "Profit generation in progress, my lord!",
    "Your trading empire expands, my lord!",
    "Market dominance activated, my lord!",
    "The money printer goes BRRR, my lord!",
]

GENERAL_QUOTES = [
    "System notification, my lord!",
    "Your trading empire grows, my lord!",
    "The machines serve you well, my lord!",
    "All systems operational, my lord!",
]

# Component emojis
COMPONENT_EMOJIS = {
    NotificationType.GENERAL: "⚡",
    NotificationType.DATA_SCRAPER: "🏯",
    NotificationType.TRAINERS: "🤖",
    NotificationType.TRADERS: "💰",
}

def get_quotes_for_type(notification_type: NotificationType) -> list:
    """Get appropriate quotes for the notification type."""
    quote_map = {
        NotificationType.DATA_SCRAPER: SCRAPER_QUOTES,
        NotificationType.TRAINERS: TRAINER_QUOTES,
        NotificationType.TRADERS: TRADER_QUOTES,
        NotificationType.GENERAL: GENERAL_QUOTES,
    }
    return quote_map.get(notification_type, GENERAL_QUOTES)

def send_discord_notification(
    message: str,
    notification_type: NotificationType = NotificationType.GENERAL,
    include_quote: bool = True
) -> bool:
    """
    Send a Discord notification to the appropriate webhook.
    
    Args:
        message: The message to send
        notification_type: Type of notification determining which webhook to use
        include_quote: Whether to include a component-specific quote
        
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    try:
        # Get the webhook URL for this notification type
        webhook_url = os.getenv(notification_type.value)
        
        # Fall back to general webhook if specific one is not configured
        if not webhook_url or webhook_url == "your_" + notification_type.value.lower() + "_here":
            webhook_url = os.getenv("DISCORD_WEBHOOK")
            if not webhook_url:
                logger.warning(f"⚠️ No Discord webhook configured for {notification_type.name}. Skipping notification.")
                return False
        
        # Prepare the message content
        emoji = COMPONENT_EMOJIS.get(notification_type, "📢")
        content = f"{emoji} **{message}**"
        
        if include_quote:
            quotes = get_quotes_for_type(notification_type)
            quote = random.choice(quotes)
            content = f"{emoji} **{quote}**\n\n{message}"
        
        # Send the notification
        payload = {"content": content}
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 204:
            logger.info(f"✅ Discord notification sent successfully for {notification_type.name}")
            return True
        else:
            logger.warning(f"❌ Failed to send Discord notification for {notification_type.name}: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception while sending Discord notification for {notification_type.name}: {e}")
        return False

def send_scraper_notification(message: str, include_quote: bool = True) -> bool:
    """Send a notification specifically for the data scraper."""
    return send_discord_notification(message, NotificationType.DATA_SCRAPER, include_quote)

def send_trainer_notification(message: str, include_quote: bool = True) -> bool:
    """Send a notification for any model trainer (RF, XGBoost, etc.)."""
    return send_discord_notification(message, NotificationType.TRAINERS, include_quote)

def send_trader_notification(message: str, include_quote: bool = True) -> bool:
    """Send a notification for trading operations."""
    return send_discord_notification(message, NotificationType.TRADERS, include_quote)

def send_general_notification(message: str, include_quote: bool = True) -> bool:
    """Send a general system notification."""
    return send_discord_notification(message, NotificationType.GENERAL, include_quote)

# For backward compatibility
def send_discord_alert(message: str) -> bool:
    """Legacy function for backward compatibility."""
    return send_general_notification(message)

# Specific trainer functions for backward compatibility
def send_rf_trainer_notification(message: str, include_quote: bool = True) -> bool:
    """Send a Random Forest trainer notification (uses general trainer webhook)."""
    return send_trainer_notification(f"🌲 **Random Forest**: {message}", include_quote)

def send_xgb_trainer_notification(message: str, include_quote: bool = True) -> bool:
    """Send an XGBoost trainer notification (uses general trainer webhook)."""
    return send_trainer_notification(f"🚀 **XGBoost**: {message}", include_quote)

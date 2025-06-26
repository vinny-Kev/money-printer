# Discord Webhook Setup Instructions

## Overview
The Money Printer trading system now supports separate Discord webhooks for different component types:
- **Data Scraper**: Notifications about data collection and market monitoring
- **Trainers**: Notifications about all model training (Random Forest, XGBoost, etc.)
- **Traders**: Notifications about trading operations and bot activities
- **General**: Fallback for system-wide notifications

## Component-Specific Notifications

### Data Scraper 🏯
- Market data collection status
- Trading pair monitoring updates
- Storage and system alerts
- **Emoji**: 🏯 **Quotes**: "Yes My Lord! I'm on it!", "The data shall be collected, my lord."

### Model Trainers 🤖
- Training pipeline status for all models
- Model performance metrics
- Training completion notifications
- **Emoji**: 🤖 **Quotes**: "Training commenced, my lord!", "The models are learning patterns, my lord!"

### Traders 💰
- Trading operations and bot activities
- Profit/loss notifications
- Market execution updates
- **Emoji**: 💰 **Quotes**: "Trading operations initiated, my lord!", "The money printer goes BRRR, my lord!"

## Setting Up Discord Webhooks

### Step 1: Create Discord Webhooks

1. **Go to your Discord server**
2. **Right-click on the channel** where you want notifications
3. **Select "Edit Channel"**
4. **Go to "Integrations" tab**
5. **Click "Create Webhook"**
6. **Name the webhook** (e.g., "Data Scraper", "Model Trainers", "Traders")
7. **Copy the webhook URL**

Create **three separate webhooks** for:
- Data Scraper notifications
- Model Trainer notifications (covers all trainers: RF, XGBoost, etc.)
- Trading Bot notifications

### Step 2: Update Environment Variables

Edit your `.env` file and replace the placeholder webhook URLs:

```env
# Discord Integration (OPTIONAL)
# Main webhook for general system notifications
DISCORD_WEBHOOK=https://discord.com/api/webhooks/your_main_webhook_here

# Separate webhooks for each component type
DISCORD_WEBHOOK_DATA_SCRAPER=https://discord.com/api/webhooks/your_data_scraper_webhook_here
DISCORD_WEBHOOK_TRAINERS=https://discord.com/api/webhooks/your_trainers_webhook_here
DISCORD_WEBHOOK_TRADERS=https://discord.com/api/webhooks/your_traders_webhook_here
```

### Step 3: Test the Notifications

Run the following commands to test each component:

```bash
# Test all notifications at once
python test_notifications.py

# Or use the main CLI
python main.py test-notifications

# Test individual components
python src/data_collector/data_scraper.py
python src/model_training/random_forest_trainer.py
python src/model_variants/xgboost_trainer.py
```

## Notification Fallback System

If a specific webhook is not configured, the system will:
1. **Try the component-specific webhook first**
2. **Fall back to the main DISCORD_WEBHOOK**
3. **Log a warning if no webhook is available**

This ensures notifications are never lost even if you only configure one webhook.

## Example Notifications

### Data Scraper
```
🏯 **Yes My Lord! I'm on it!**

🚀 **Data Scraper Started**: Now monitoring 100 trading pairs
```

### Model Trainers (Random Forest)
```
🤖 **Training commenced, my lord!**

🌲 **Random Forest**: 🎯 **Training Complete!**

📊 **Final Results:**
• Test Accuracy: 0.8542
• Test F1 Score: 0.8234
• Test AUC-ROC: 0.8967
• Model saved to: data/models/random_forest/trained_model.pkl

🌲 The Random Forest is ready for battle!
```

### Model Trainers (XGBoost)
```
🤖 **The models are learning patterns, my lord!**

🚀 **XGBoost**: 🎯 **Training Complete!**

📊 **Final Results:**
• Test Accuracy: 0.8734
• Test F1 Score: 0.8456
• Test AUC-ROC: 0.9012
• Model saved to: data/models/xgboost/trained_model.pkl

🚀 The XGBoost ensemble is ready for deployment!
```

### Trading Bots
```
💰 **Trading operations initiated, my lord!**

💰 **Trade Executed**: BUY BTCUSDT - $100 USDT
💰 **Profit Generated**: +$5.50 USDT (+5.5%)
💰 **The money printer goes BRRR, my lord!**
```

## Files Updated

The following files have been updated to support the new notification system:

1. **`.env`** - Added separate webhook environment variables
2. **`src/config.py`** - Added webhook configuration loading
3. **`src/discord_notifications.py`** - New centralized notification system
4. **`src/data_collector/data_scraper.py`** - Updated to use scraper notifications
5. **`src/model_training/random_forest_trainer.py`** - Added RF trainer notifications
6. **`src/model_variants/xgboost_trainer.py`** - Added XGBoost trainer notifications

## Usage in Your Code

To use the notification system in your own code:

```python
from discord_notifications import (
    send_scraper_notification,
    send_trainer_notification, 
    send_trader_notification,
    send_general_notification
)

# Send specific notifications
send_scraper_notification("Data collection started")
send_trainer_notification("Model training completed")
send_trader_notification("Trade executed successfully")
send_general_notification("System update")

# Backward compatibility - these will use the trainer webhook
from discord_notifications import send_rf_trainer_notification, send_xgb_trainer_notification
send_rf_trainer_notification("Random Forest training started")  # Goes to TRAINERS webhook
send_xgb_trainer_notification("XGBoost training completed")     # Goes to TRAINERS webhook
```

## Next Steps

1. **Create your Discord webhooks** following Step 1 above
2. **Update your `.env` file** with the actual webhook URLs
3. **Test the notifications** by running each component
4. **Enjoy real-time updates** from your trading system!

The system is now fully configured for local storage with separate Discord notifications for each major component. Happy trading! 🚀

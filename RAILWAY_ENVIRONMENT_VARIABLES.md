# 🚀 Railway Deployment Environment Variables

## Required Environment Variables for Railway

Copy these environment variables to your Railway project dashboard:

### Core Application Settings
```
PYTHONPATH=/app:/app/src
ENVIRONMENT=production
FORCE_REAL_MODULES=true
PORT=8000
LOG_LEVEL=INFO
```

### Discord Bot Configuration
```
DISCORD_BOT_TOKEN=your_discord_bot_token_here
DISCORD_WEBHOOK=your_general_webhook_url_here
DISCORD_WEBHOOK_DATA_SCRAPER=your_scraper_webhook_url_here
DISCORD_WEBHOOK_TRAINERS=your_trainers_webhook_url_here
DISCORD_WEBHOOK_TRADERS=your_traders_webhook_url_here
DISCORD_CHANNEL_ID=your_channel_id_here
```

### Binance Trading API
```
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_API_KEY_TESTNET=your_testnet_api_key_here
BINANCE_SECRET_KEY_TESTNET=your_testnet_secret_key_here
```

### Railway Integration (Optional)
```
RAILWAY_API_TOKEN=your_railway_token_here
RAILWAY_PROJECT_ID=your_project_id_here
```

## How to Set Environment Variables in Railway

### Method 1: Railway Dashboard
1. Go to your Railway project dashboard
2. Click on your service
3. Go to "Variables" tab
4. Click "New Variable"
5. Add each variable name and value
6. Click "Deploy" to apply changes

### Method 2: Railway CLI
```bash
railway variables set DISCORD_BOT_TOKEN=your_token_here
railway variables set BINANCE_API_KEY=your_key_here
# ... add all variables
railway deploy
```

## 🔐 Security Notes

### For Testing/Development:
- Use Binance **TESTNET** API keys initially
- Start with paper trading mode
- Use Discord webhooks for testing channels

### For Production:
- Use real Binance API keys with **READ and TRADE** permissions only
- Restrict API keys to specific IP addresses if possible
- Use dedicated Discord channels for production alerts
- Enable 2FA on all accounts

## 📋 Environment Variables Checklist

### Required for Basic Operation:
- [x] `DISCORD_BOT_TOKEN` - Discord bot token
- [x] `DISCORD_WEBHOOK` - General notifications webhook
- [x] `PYTHONPATH` - Python module path
- [x] `ENVIRONMENT` - Set to "production"
- [x] `FORCE_REAL_MODULES` - Set to "true"

### Required for Trading:
- [x] `BINANCE_API_KEY` - Binance API key
- [x] `BINANCE_SECRET_KEY` - Binance secret key
- [x] `BINANCE_API_KEY_TESTNET` - Testnet API key (for testing)
- [x] `BINANCE_SECRET_KEY_TESTNET` - Testnet secret key (for testing)

### Optional but Recommended:
- [ ] `DISCORD_WEBHOOK_DATA_SCRAPER` - Scraper notifications
- [ ] `DISCORD_WEBHOOK_TRAINERS` - Model training notifications
- [ ] `DISCORD_WEBHOOK_TRADERS` - Trading notifications
- [ ] `LOG_LEVEL` - Set to "INFO" or "DEBUG"
- [ ] `PORT` - Set to "8000" (Railway handles this automatically)

## 🚨 Common Issues and Solutions

### Issue: "Environment variable not found"
**Solution**: Make sure the variable name is exactly as shown above (case-sensitive)

### Issue: "Discord bot not responding"
**Solution**: Check that `DISCORD_BOT_TOKEN` is set correctly and the bot is invited to your server

### Issue: "Trading API errors"
**Solution**: Verify Binance API keys have the correct permissions and are not expired

### Issue: "Module import errors"
**Solution**: Ensure `PYTHONPATH=/app:/app/src` is set correctly

## 🔄 Redeployment Process

After setting environment variables:

1. **Railway will automatically redeploy** when you add/change variables
2. **Or manually trigger redeploy**: Click "Deploy" in Railway dashboard
3. **Check logs**: Monitor the deployment logs for any errors
4. **Test the bot**: Send `/ping` command in Discord to verify it's working

## ✅ Verification Commands

Once deployed, test these Discord commands:
- `/ping` - Check bot responsiveness
- `/status` - Check system status
- `/stats` - View comprehensive statistics
- `/balance` - Check account balance (if trading APIs configured)

## 📞 Support

If you encounter issues:
1. Check Railway deployment logs
2. Verify all environment variables are set correctly
3. Test with testnet APIs first
4. Use `/status` command to diagnose issues

---

*Updated: June 27, 2025*
*For Money Printer Trading Bot v1.0*

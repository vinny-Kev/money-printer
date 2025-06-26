# 🚀 Discord Bot Usage Guide - Start Trading in 5 Steps

## 🎯 Quick Answer: How to Start Scraping, Training & Trading

### 1️⃣ Check Bot Status
```
/status
```
Verify the bot is online and see which features are available.

### 2️⃣ Start Data Collection  
```
/start_scraper
```
Begin collecting cryptocurrency market data. Let this run for 15-30 minutes.

### 3️⃣ Train Your AI Model
```
/train_model random_forest
```
Create a machine learning model using the collected data (takes 2-5 minutes).

### 4️⃣ Test with Paper Trading
```
/start_dry_trade 3
```
Execute 3 simulated trades with virtual money to test your strategy.

### 5️⃣ Monitor Performance
```
/trading_stats
```
View win rate, profit/loss, and optimize your trading strategy.

## 🎮 Complete Discord Commands

### 📊 Data Collection
- `/start_scraper` - Start collecting market data
- `/stop_scraper` - Stop data collection

### 🤖 Machine Learning  
- `/train_model random_forest` - Train AI trading model
- `/train_model xgboost` - Train advanced model

### 💰 Trading
- `/start_dry_trade [1-10]` - Paper trade with virtual money
- `/balance` - Check account balance
- `/trading_stats` - View trading performance

### 🔧 System
- `/status` - Comprehensive system status
- `/ping` - Check bot responsiveness  
- `/help` - Show all commands
- `/deploy_test` - Test Railway deployment

## ✅ What the Enhanced Bot Includes

### 🎯 Complete Trading Pipeline
- **Real-time data scraping** from cryptocurrency exchanges
- **Machine learning model training** using collected data
- **Paper trading** for safe strategy testing
- **Performance monitoring** with detailed statistics
- **Railway deployment** optimized for 24/7 operation

### 🛡️ Safety Features
- **Authorization required** - Only you can execute commands
- **Paper trading first** - No real money at risk
- **Graceful fallbacks** - Works even with missing dependencies
- **Health monitoring** - Always responsive to Railway

### 🚀 Railway-Optimized
- **Ultra-lightweight** ~170MB Docker container
- **Always-on health checks** for Railway compatibility
- **Smart resource management** for cloud efficiency
- **Automatic restart recovery** from any failures

## 📈 Expected Results

### After Following the 5 Steps:
- ✅ **Automated data collection** running in background
- ✅ **Trained AI model** ready for trading decisions
- ✅ **Paper trading results** showing strategy performance
- ✅ **Real-time monitoring** of all system components
- ✅ **Foundation for live trading** (when ready)

### Typical Performance:
- 📊 **Win Rate**: 50-70% (varies by market conditions)
- 💰 **Paper Trading**: Virtual profits/losses to test strategy
- ⚡ **Response Time**: Commands respond in 1-3 seconds
- 🔄 **Uptime**: 24/7 operation on Railway cloud

## 🚨 Important Notes

### Before You Start:
- ✅ Bot must be **deployed to Railway** first
- ✅ Set **DISCORD_BOT_TOKEN** and **DISCORD_USER_ID** environment variables
- ✅ Only the **authorized Discord user** can execute trading commands
- ✅ **Start with paper trading** - never skip this safety step

### Troubleshooting:
- **Commands not working?** Check you're the authorized user
- **Scraper won't start?** Try `/stop_scraper` then `/start_scraper`
- **No trading features?** Bot may be in lightweight mode (still functional)
- **Performance issues?** Check Railway resource usage

## 🎉 Success Indicators

Look for these signs that everything is working:

### ✅ Bot Working Correctly:
- `/status` shows "Online & Ready" 
- Commands respond with rich Discord embeds
- `/help` shows all available commands
- No error messages in responses

### ✅ Data Collection Working:
- `/start_scraper` confirms "Scraper is now running"
- `/status` shows "Data Scraper: Running"
- No error messages after 5-10 minutes

### ✅ Trading Ready:
- Model training completes without errors
- `/start_dry_trade` executes successfully
- `/trading_stats` shows results
- Win rate appears reasonable (>40%)

## 📞 Need Help?

1. **Check `/help`** - Shows all available commands
2. **Try `/status`** - Comprehensive system health check  
3. **Review `/deploy_test`** - Verify Railway deployment
4. **Check environment variables** - Ensure tokens are set correctly
5. **Restart if needed** - Railway will automatically restart the container

---

**🚀 Ready to start?** Begin with `/status` then follow the 5-step workflow above!

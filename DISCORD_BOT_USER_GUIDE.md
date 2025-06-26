# 🤖 Discord Trading Bot User Guide

## Overview
This guide walks you through using the enhanced Discord trading bot to scrape data, train models, and execute trades through Discord commands.

## 🚀 Quick Start Workflow

### Step 1: Verify Bot Status
Start by checking if the bot is online and what features are available:

```
/status
```

This shows:
- ✅ Bot connectivity and latency
- 🔧 Available systems (Trading, Scraping, Model Training)
- 🔄 Current operations status

### Step 2: Start Data Collection
Before trading, you need market data. Start the scraper:

```
/start_scraper
```

**What this does:**
- 📊 Collects real-time cryptocurrency market data
- 💾 Stores data locally for model training
- 🔄 Runs continuously in the background
- ⚡ Uses minimal resources (Railway-optimized)

**Expected output:** Confirmation that scraper is running with a fun peasant quote!

### Step 3: Train Your First Model
Once you have some data (wait 5-10 minutes), train a trading model:

```
/train_model random_forest
```

**What this does:**
- 🤖 Creates a machine learning model using collected data
- 📈 Learns patterns from historical price movements
- 💾 Saves the trained model for trading
- ⚡ Takes 2-5 minutes depending on data volume

**Available model types:**
- `random_forest` (recommended for beginners)
- `xgboost` (advanced, more resource intensive)

### Step 4: Test with Paper Trading
Before risking real money, test your strategy:

```
/start_dry_trade 3
```

**What this does:**
- 📝 Simulates trading with fake money
- 💰 Uses your trained models to make decisions
- 📊 Shows real-time results for each trade
- 🎯 Helps you evaluate model performance

**Parameters:**
- Number of trades: 1-10 (start with 1-3 for testing)

### Step 5: Monitor Performance
Track your trading results:

```
/trading_stats
```

**Shows:**
- 📊 Total trades executed
- 🎯 Win rate percentage
- 💰 Total profit/loss
- 📈 Best and worst trades
- 📊 Average trade performance

## 📚 Complete Command Reference

### 🔧 Basic Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/ping` | Check bot responsiveness | `/ping` |
| `/status` | Comprehensive system status | `/status` |
| `/help` | Show all available commands | `/help` |
| `/deploy_test` | Test Railway deployment | `/deploy_test` |

### 📊 Data Collection Commands

| Command | Description | When to Use |
|---------|-------------|-------------|
| `/start_scraper` | Start data collection | Before training models |
| `/stop_scraper` | Stop data collection | To save resources |

### 🤖 Machine Learning Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/train_model [type]` | Train a new model | `/train_model random_forest` |

### 💰 Trading Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/start_dry_trade [num]` | Paper trading | `/start_dry_trade 5` |
| `/balance` | Check USDT balance | `/balance` |
| `/trading_stats` | View performance | `/trading_stats` |

## 🎯 Best Practices

### Data Collection
- ✅ **Start scraper early:** Let it run for at least 30 minutes before training
- ✅ **Monitor resources:** Check `/status` regularly
- ✅ **Stop when not needed:** Use `/stop_scraper` to save Railway resources

### Model Training
- ✅ **Start with random_forest:** It's faster and more reliable
- ✅ **Wait for sufficient data:** At least 100+ data points
- ✅ **Retrain regularly:** Every few hours for better performance

### Trading
- ✅ **Always start with paper trading:** Never skip this step
- ✅ **Test small batches:** Start with 1-3 trades
- ✅ **Monitor win rate:** Aim for >60% before live trading
- ✅ **Check balance regularly:** Use `/balance` command

## 🚨 Troubleshooting

### "System not available" errors
**Cause:** The bot is in lightweight mode or missing dependencies
**Solution:** 
1. Check `/deploy_test` for system capabilities
2. This is expected in minimal Railway deployments
3. The bot will gracefully handle missing features

### Scraper won't start
**Cause:** Resource limitations or missing data directory
**Solution:**
1. Check `/status` for current operations
2. Try `/stop_scraper` then `/start_scraper`
3. Railway may have restarted the container

### Trading commands fail
**Cause:** No trained models or insufficient data
**Solution:**
1. Ensure `/start_scraper` has been running
2. Train a model with `/train_model`
3. Wait for training to complete

### Permission denied
**Cause:** Not authorized to use the bot
**Solution:**
- Only the configured Discord user can use trading commands
- Check with the bot administrator

## 🌟 Advanced Features

### Continuous Operation
The bot is designed to run 24/7 on Railway:
- 🔄 Automatic restarts on failures
- 💓 Health checks keep it running
- 📊 Background data collection
- ⚡ Resource-optimized for cloud deployment

### System Monitoring
- Use `/status` for real-time system health
- Monitor Railway dashboard for resource usage
- Check Discord notifications for important updates

### Multi-Model Trading
- Train multiple model types
- Compare performance with `/trading_stats`
- The system automatically uses the best performing model

## 🎉 Success Indicators

### Data Collection Success
- ✅ Scraper shows "Running" in `/status`
- ✅ No error messages after starting
- ✅ Regular activity in Railway logs

### Training Success
- ✅ Training completes without errors
- ✅ Model files are created (check logs)
- ✅ `/trading_stats` shows available models

### Trading Success
- ✅ Dry trades execute successfully
- ✅ Win rate >50% consistently
- ✅ Positive total PnL over time

## 📞 Support

If you encounter issues:
1. Check `/status` for system health
2. Review this guide for common solutions
3. Check Railway deployment logs
4. Ensure all environment variables are set
5. Try restarting with Railway deployment reset

---

**Remember:** Always start with paper trading and never invest more than you can afford to lose!

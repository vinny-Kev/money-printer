# 🚀 User Guide - AI Trading Bot with Discord Control

## 🎯 Quick Start - Get Trading in 5 Minutes

### Prerequisites
- Discord bot token
- Discord user ID
- (Optional) Binance API credentials for live trading

### 1. Deploy to Railway
1. Fork this repository to your GitHub
2. Connect to Railway and deploy
3. Set environment variables:
   ```env
   DISCORD_BOT_TOKEN=your_discord_bot_token
   DISCORD_USER_ID=your_discord_user_id
   ```

### 2. Start Trading Through Discord

#### Step 1: Check System Status
```
/status
```
Verify bot is online and see available features.

#### Step 2: Start Data Collection
```
/start_scraper
```
Begin collecting cryptocurrency market data (let run 15-30 minutes).

#### Step 3: Train AI Model
```
/train_model random_forest
```
Create machine learning model using collected data (2-5 minutes).

#### Step 4: Test with Paper Trading
```
/start_dry_trade 3
```
Execute 3 simulated trades with virtual money (safe testing).

#### Step 5: Monitor Performance
```
/trading_stats
```
View win rate, profit/loss, and optimize strategy.

## 📱 Complete Discord Commands

### 📊 Data Collection
| Command | Description | Example |
|---------|-------------|---------|
| `/start_scraper` | Start collecting market data | `/start_scraper` |
| `/stop_scraper` | Stop data collection | `/stop_scraper` |

### 🤖 Machine Learning
| Command | Description | Example |
|---------|-------------|---------|
| `/train_model [type]` | Train trading models | `/train_model random_forest` |

### 💰 Trading
| Command | Description | Example |
|---------|-------------|---------|
| `/start_dry_trade [num]` | Paper trading (1-10 trades) | `/start_dry_trade 5` |
| `/balance` | Check USDT balance | `/balance` |
| `/trading_stats` | View performance stats | `/trading_stats` |

### 🔧 System
| Command | Description | Example |
|---------|-------------|---------|
| `/status` | System health check | `/status` |
| `/ping` | Bot responsiveness | `/ping` |
| `/help` | Show all commands | `/help` |
| `/deploy_test` | Test Railway deployment | `/deploy_test` |

## 🎯 Trading Workflow

### 🟢 First Time Setup
1. **Deploy bot** to Railway with environment variables
2. **Test connection** with `/status` command
3. **Verify permissions** (only you can execute trading commands)

### 🔄 Daily Trading Routine
1. **Start scraper**: `/start_scraper` (morning)
2. **Train model**: `/train_model random_forest` (after data collection)
3. **Paper trade**: `/start_dry_trade 3` (test strategy)
4. **Monitor results**: `/trading_stats` (track performance)
5. **Stop scraper**: `/stop_scraper` (evening to save resources)

### 📈 Performance Optimization
- **Monitor win rate** with `/trading_stats`
- **Retrain models** when performance drops
- **Adjust trade frequency** based on market conditions
- **Scale up gradually** after consistent profits

## 🛡️ Safety Features

### 🔒 Built-in Protections
- ✅ **Paper trading first** - No real money at risk initially
- ✅ **Authorization required** - Only you can execute commands
- ✅ **Resource monitoring** - Automatic system health checks
- ✅ **Graceful failures** - System continues even with errors

### ⚠️ Important Safety Rules
- 🚨 **Always start with paper trading**
- 🚨 **Never skip the testing phase**
- 🚨 **Monitor performance regularly**
- 🚨 **Start with small amounts for live trading**

## 🚨 Troubleshooting

### Bot Commands Not Working
**Problem**: Commands return "not authorized" or don't respond
**Solution**: 
- Verify `DISCORD_USER_ID` matches your Discord ID
- Check bot has proper permissions in Discord server
- Try `/deploy_test` to verify deployment

### Scraper Won't Start
**Problem**: `/start_scraper` fails or shows errors
**Solution**:
- Check `/status` for current operations
- Try `/stop_scraper` then `/start_scraper`
- Railway may have restarted (normal behavior)

### Trading Commands Fail
**Problem**: Paper trading doesn't execute
**Solution**:
- Ensure scraper has been running for 15+ minutes
- Train a model first with `/train_model`
- Check `/status` to verify system readiness

### Performance Issues
**Problem**: Slow responses or timeouts
**Solution**:
- Check Railway resource usage in dashboard
- Monitor with `/status` command regularly
- Restart deployment if needed

## 📊 Expected Results

### ✅ Successful Setup Indicators
- Bot responds to `/status` with system information
- Commands execute without authorization errors
- `/help` shows complete command list
- Railway health checks are passing

### ✅ Data Collection Success
- `/start_scraper` confirms "Scraper is now running"
- `/status` shows "Data Scraper: Running"
- No error messages after 10-15 minutes

### ✅ Trading Readiness
- Model training completes without errors
- `/start_dry_trade` executes successfully
- `/trading_stats` shows trading results
- Win rate appears reasonable (>40%)

### 📈 Performance Targets
- **Win Rate**: Aim for 55-70%
- **Response Time**: Commands respond in 1-3 seconds
- **Uptime**: 24/7 operation on Railway
- **Resource Usage**: <300MB memory typically

## 🎯 Advanced Tips

### 🔧 Optimization Strategies
- **Best Training Time**: After 30+ minutes of data collection
- **Model Selection**: Start with `random_forest`, try `xgboost` for advanced
- **Trade Frequency**: Begin with 1-3 trades, scale up gradually
- **Market Timing**: Avoid training during high volatility periods

### 🌟 Pro Features
- **Continuous Operation**: Let scraper run 24/7 for best data
- **Regular Retraining**: Train new models daily for optimal performance
- **Performance Tracking**: Use `/trading_stats` to identify trends
- **Resource Management**: Monitor Railway usage to avoid limits

## 📞 Support

### 🆘 Getting Help
1. Check `/help` command for available options
2. Use `/status` for comprehensive system health
3. Review this guide for common solutions
4. Verify environment variables in Railway
5. Check Railway logs for detailed error information

### 🐛 Reporting Issues
- Include output from `/status` and `/deploy_test`
- Provide recent command history
- Check Railway deployment logs
- Note any error messages received

---

## 🎉 Success!

Once everything is working, you'll have:
- ✅ **24/7 automated data collection**
- ✅ **AI-powered trading models**
- ✅ **Paper trading for safe testing**
- ✅ **Real-time performance monitoring**
- ✅ **Complete Discord control interface**

**Ready to start?** Begin with `/status` and follow the 5-step workflow above!

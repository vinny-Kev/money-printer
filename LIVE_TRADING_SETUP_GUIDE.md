# 🚀 Live Trading Setup Guide

## Current Issue: Balance Shows "Unable to fetch"

The trading bot is currently in **Paper Trading Mode**, which is why it shows "Unable to fetch" for the balance instead of your real Binance balance.

## ✅ Solution: Enable Live Trading

### Step 1: Set Railway Environment Variable

You need to add the `LIVE_TRADING` environment variable to your Railway deployment:

1. **Go to Railway Dashboard**
   - Visit [railway.app](https://railway.app)
   - Navigate to your Money Printer project
   - Click on your deployed service

2. **Add Environment Variable**
   - Go to "Variables" tab
   - Click "New Variable"
   - Set: `LIVE_TRADING` = `true`
   - Click "Add"

3. **Deploy Changes**
   - Click "Deploy" to apply the changes
   - Wait for deployment to complete

### Step 2: Verify API Keys Are Set

Make sure these environment variables are properly set in Railway:

```
BINANCE_API_KEY=your_real_binance_api_key
BINANCE_SECRET_KEY=your_real_binance_secret_key
LIVE_TRADING=true
```

## ⚠️ Important Warnings

### 🔴 LIVE TRADING RISKS
Once you enable `LIVE_TRADING=true`, the bot will:
- ✅ Display your **real Binance balance**
- ✅ Execute **real trades** with real money
- ✅ Place **actual buy/sell orders** on Binance
- ⚠️ Use your **real USDT** for trading

### 🛡️ Safety Features Active
The bot has comprehensive safety mechanisms:
- Daily trade limits (max 50 trades/day)
- Hourly trade limits (max 10 trades/hour)
- Position sizing based on account balance
- Stop-loss protection
- Volatility filters
- Minimum balance checks

### 💰 Recommended for Live Trading
- **Minimum Balance**: $100+ USDT for meaningful trades
- **Start Small**: Consider starting with $50-200 to test
- **Monitor Closely**: Watch the Discord notifications
- **Have Stop-Loss**: All trades include automatic stop-loss

## 📊 What Changes After Enabling Live Trading

### Before (Paper Trading):
```
💰 Current Balance: $1000 (Paper)
📊 All trades are simulated
🔄 No real money involved
```

### After (Live Trading):
```
💰 Current Balance: $X.XX USDT (Real Binance Balance)
📊 All trades use real money
💸 Real profits and losses
```

## 🔧 Testing the Setup

1. **Set Environment Variable**: `LIVE_TRADING=true`
2. **Wait for Deployment**: ~2-3 minutes
3. **Check Bot Status**: Use `/status` command in Discord
4. **Verify Balance**: Should show your real USDT balance

## 🆘 If Something Goes Wrong

### Balance Still Shows "Unable to fetch"
1. Check Railway logs for API errors
2. Verify API keys are correct
3. Ensure API keys have trading permissions
4. Check Binance API restrictions

### Bot Shows Errors
1. Check Discord notifications for details
2. Review Railway deployment logs
3. Ensure all environment variables are set

## 📞 Emergency Stop

If you need to immediately stop live trading:
1. **Set** `LIVE_TRADING=false` in Railway
2. **Redeploy** the service
3. **Monitor** any open positions manually in Binance

## 🎯 Next Steps

1. Set `LIVE_TRADING=true` in Railway
2. Verify real balance appears in Discord
3. Start with small test trades
4. Monitor performance and adjust as needed

---

**Remember**: Live trading involves real financial risk. Only trade with money you can afford to lose, and always monitor the bot's performance closely.

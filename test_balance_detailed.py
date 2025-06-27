#!/usr/bin/env python3
"""
Direct Binance Balance Test - Check exactly what's in your account
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_binance_detailed():
    """Test Binance account in detail."""
    
    logger.info("💰 DETAILED Binance Account Analysis...")
    
    try:
        from binance.client import Client
        
        api_key = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        if not api_key or not secret_key:
            logger.error("❌ Missing Binance API credentials!")
            return False
        
        logger.info(f"🔑 Using API Key: {api_key[:10]}...")
        
        # Create client (make sure it's NOT testnet)
        client = Client(api_key, secret_key, testnet=False)
        
        # Test connection
        logger.info("🔄 Testing connection...")
        server_time = client.get_server_time()
        logger.info(f"✅ Connected! Server time: {server_time}")
        
        # Get account information
        logger.info("📊 Getting account information...")
        account = client.get_account()
        
        logger.info(f"📈 Account Type: {account.get('accountType', 'Unknown')}")
        logger.info(f"🔄 Can Trade: {account.get('canTrade', False)}")
        logger.info(f"🔄 Can Withdraw: {account.get('canWithdraw', False)}")
        logger.info(f"🔄 Can Deposit: {account.get('canDeposit', False)}")
        
        # Check ALL balances
        logger.info("\n💰 ALL ACCOUNT BALANCES:")
        total_balances = 0
        usdt_balance = 0
        
        for balance in account['balances']:
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_balance = free_balance + locked_balance
            
            if total_balance > 0:
                logger.info(f"  💵 {balance['asset']}: Free={free_balance:.8f}, Locked={locked_balance:.8f}, Total={total_balance:.8f}")
                total_balances += 1
                
                if balance['asset'] == 'USDT':
                    usdt_balance = total_balance
        
        if total_balances == 0:
            logger.warning("❌ NO BALANCES FOUND! Account appears empty.")
        else:
            logger.info(f"\n📊 Summary: Found {total_balances} assets with balances")
        
        # Focus on USDT
        logger.info(f"\n🎯 USDT Analysis:")
        logger.info(f"   💰 USDT Balance: {usdt_balance:.8f}")
        logger.info(f"   💵 USD Value: ~${usdt_balance:.2f}")
        
        # Check if this matches expected 500 PHP deposit
        expected_usd = 500 / 58  # 500 PHP to USD
        logger.info(f"   📊 Expected from 500 PHP: ~${expected_usd:.2f}")
        
        if usdt_balance >= expected_usd * 0.8:  # Allow 20% variance
            logger.info("   ✅ Balance matches expected deposit!")
        else:
            logger.warning("   ⚠️ Balance doesn't match expected amount")
        
        # Check deposit history
        logger.info("\n📥 Checking recent deposits...")
        try:
            deposits = client.get_deposit_history(limit=10)
            if deposits:
                logger.info("Recent deposits:")
                for deposit in deposits:
                    status = deposit.get('status', 'Unknown')
                    amount = deposit.get('amount', '0')
                    coin = deposit.get('coin', 'Unknown')
                    insert_time = deposit.get('insertTime', 'Unknown')
                    logger.info(f"   📥 {coin}: {amount} - Status: {status} - Time: {insert_time}")
            else:
                logger.warning("   ❌ No recent deposits found")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not fetch deposit history: {e}")
        
        # Test trading permissions
        logger.info("\n🛡️ Testing trading permissions...")
        try:
            # Try to get trading fees (this tests trading API access)
            trade_fee = client.get_trade_fee(symbol='BTCUSDT')
            logger.info("   ✅ Trading API access confirmed")
        except Exception as e:
            logger.warning(f"   ⚠️ Trading API issue: {e}")
        
        return usdt_balance > 0
        
    except Exception as e:
        logger.error(f"❌ Binance test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_discord_balance_direct():
    """Test the Discord balance function directly."""
    
    logger.info("🤖 Testing Discord Balance Function...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        # Try importing the trading module
        from trading_bot.trade_runner import get_account_balance_safe
        
        logger.info("🔄 Testing get_account_balance_safe()...")
        balance_info = get_account_balance_safe()
        
        logger.info(f"📊 Balance function result: {balance_info}")
        
        if balance_info["status"] == "success":
            logger.info(f"✅ Balance: {balance_info['balance']:.4f} USDT")
            logger.info(f"📊 Mode: {balance_info['mode']}")
            
            if balance_info["mode"] == "paper":
                logger.warning("⚠️ Currently in PAPER TRADING mode!")
                logger.info("💡 Set LIVE_TRADING=true to use real balance")
            
            return True
        else:
            logger.error(f"❌ Balance function failed: {balance_info}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Discord balance test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all balance tests."""
    
    logger.info("🚀 Starting COMPREHENSIVE Balance Analysis...")
    logger.info("="*60)
    
    # Test 1: Direct Binance API
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 1: Direct Binance API Access")
    logger.info("="*60)
    binance_success = test_binance_detailed()
    
    # Test 2: Discord Balance Function
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 2: Discord Balance Function")
    logger.info("="*60)
    discord_success = test_discord_balance_direct()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 BALANCE TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"  Direct Binance API: {'✅ PASSED' if binance_success else '❌ FAILED'}")
    logger.info(f"  Discord Balance Function: {'✅ PASSED' if discord_success else '❌ FAILED'}")
    
    if binance_success and discord_success:
        logger.info("\n🎉 ALL TESTS PASSED! Balance detection is working.")
    else:
        logger.info("\n🔧 TROUBLESHOOTING STEPS:")
        if not binance_success:
            logger.info("  1. Verify your 500 PHP deposit completed successfully")
            logger.info("  2. Check if funds are in Spot wallet (not Futures/Margin)")
            logger.info("  3. Wait for deposit confirmation (can take 10-30 minutes)")
            logger.info("  4. Check Binance app/website to confirm balance")
        if not discord_success:
            logger.info("  5. Check if LIVE_TRADING environment variable is set correctly")
            logger.info("  6. Restart the Discord bot to reload settings")

if __name__ == "__main__":
    main()

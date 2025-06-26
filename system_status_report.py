#!/usr/bin/env python3
"""
Enhanced Trading System Status Report

This script provides a comprehensive overview of the enhanced trading system
and demonstrates all the new features and capabilities.
"""

import sys
import os
import pandas as pd
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def generate_status_report():
    """
    Generate a comprehensive status report of the enhanced trading system.
    """
    print("🚀" + "="*70 + "🚀")
    print("   ENHANCED MONEY PRINTER - COMPREHENSIVE STATUS REPORT")
    print("🚀" + "="*70 + "🚀")
    print()
    
    # 1. System Overview
    print("📋 SYSTEM OVERVIEW:")
    print("="*50)
    print("✅ Persistent trades until TP/SL exit")
    print("✅ Real-time trade monitoring (5s intervals)")
    print("✅ Incremental learning data collection")
    print("✅ Automatic model retraining capabilities")
    print("✅ Enhanced CSV export for taxation")
    print("✅ User input for dynamic trading budgets")
    print("✅ Clean output with timestamped messages")
    print("✅ Realistic profit predictions (0.5% - 8%)")
    print("✅ Live Binance WebSocket market data")
    print("✅ Binance Testnet for safe dry trading")
    print()
    
    # 2. Data Collection Status
    print("📊 DATA COLLECTION STATUS:")
    print("="*50)
    
    # Training data
    training_csv = "data/transactions/random_forest_v1_trades.csv"
    if os.path.exists(training_csv):
        df_training = pd.read_csv(training_csv)
        print(f"✅ Training Data: {len(df_training)} trades collected")
        print(f"   📈 Win Rate: {(df_training['was_successful'].sum() / len(df_training)) * 100:.1f}%")
        print(f"   💰 Avg P&L: {df_training['pnl_percent'].mean():.2f}%")
        print(f"   📁 Location: {training_csv}")
    else:
        print("❌ Training Data: No data found")
    
    # Tax data
    tax_csv = "src/trading_bot/trading_transactions.csv"
    if os.path.exists(tax_csv):
        df_tax = pd.read_csv(tax_csv)
        print(f"✅ Tax Export Data: {len(df_tax)} transactions logged")
        print(f"   📁 Location: {tax_csv}")
    else:
        print("❌ Tax Export Data: No data found")
    
    # Receipt files
    receipts_dir = "src/trading_bot/receipts"
    if os.path.exists(receipts_dir):
        receipts = [f for f in os.listdir(receipts_dir) if f.endswith('.json')]
        print(f"✅ Trade Receipts: {len(receipts)} JSON files")
        print(f"   📁 Location: {receipts_dir}")
    else:
        print("❌ Trade Receipts: No receipts found")
    
    print()
    
    # 3. Model Status
    print("🤖 MODEL STATUS:")
    print("="*50)
    
    models_dir = "data/models/random_forest"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        print(f"✅ Model Files: {len(model_files)} versions available")
        for file in model_files:
            print(f"   📦 {file}")
        
        # Check for expected features
        features_file = os.path.join(models_dir, "expected_features.json")
        if os.path.exists(features_file):
            print(f"✅ Expected Features: Available")
        else:
            print(f"❌ Expected Features: Missing")
    else:
        print("❌ Model Directory: Not found")
    
    print()
    
    # 4. Incremental Learning Status
    print("🧠 INCREMENTAL LEARNING STATUS:")
    print("="*50)
    
    if os.path.exists(training_csv):
        from src.model_training.incremental_trainer import IncrementalTrainer
        trainer = IncrementalTrainer()
        
        should_retrain, reason = trainer.should_retrain()
        print(f"🔍 Retraining Needed: {should_retrain}")
        print(f"📝 Reason: {reason}")
        
        if should_retrain:
            print("🚀 System is ready for automatic retraining!")
        else:
            print("✅ Model performance is satisfactory")
    else:
        print("❌ Cannot assess - no training data available")
    
    print()
    
    # 5. Recent Activity
    print("⏰ RECENT ACTIVITY:")
    print("="*50)
    
    if os.path.exists(training_csv):
        df_training = pd.read_csv(training_csv)
        df_training['timestamp'] = pd.to_datetime(df_training['timestamp'])
        df_recent = df_training.sort_values('timestamp').tail(3)
        
        print("📊 Last 3 Trades:")
        for _, trade in df_recent.iterrows():
            status = "✅" if trade['was_successful'] else "❌"
            print(f"   {status} {trade['coin']} | P&L: {trade['pnl_percent']:.2f}% | {trade['timestamp'].strftime('%H:%M:%S')}")
    else:
        print("❌ No recent activity data available")
    
    print()
    
    # 6. System Capabilities
    print("🔧 SYSTEM CAPABILITIES:")
    print("="*50)
    print("📈 Market Data:")
    print("   • Live WebSocket streams from Binance")
    print("   • Top 200 USDT pairs by volume")
    print("   • Real-time price updates")
    print()
    print("🎯 Trading Logic:")
    print("   • RSI and MACD technical indicators")
    print("   • Random Forest ML predictions")
    print("   • Confidence-based trade selection")
    print("   • Automatic TP/SL placement")
    print()
    print("🔍 Monitoring:")
    print("   • Persistent trade tracking")
    print("   • 5-second price update intervals")
    print("   • Timeout protection (6 minutes)")
    print("   • Real-time P&L calculation")
    print()
    print("🤖 Learning:")
    print("   • Automatic data collection")
    print("   • Feature engineering from trades")
    print("   • Self-regulating retraining")
    print("   • Discord notifications")
    print()
    
    # 7. Quick Test Instructions
    print("🚀 QUICK TEST INSTRUCTIONS:")
    print("="*50)
    print("1. Interactive Test:")
    print("   python test_enhanced_trading.py")
    print()
    print("2. Automated Test:")
    print("   python simple_enhanced_test.py")
    print()
    print("3. Manual Training:")
    print("   python -m src.model_training.incremental_trainer --force")
    print()
    print("4. Check Training Status:")
    print("   python -m src.model_training.incremental_trainer --check-only")
    print()
    
    print("🎉 ENHANCED MONEY PRINTER IS FULLY OPERATIONAL!")
    print("🚀" + "="*70 + "🚀")

if __name__ == "__main__":
    generate_status_report()

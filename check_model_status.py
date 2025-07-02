#!/usr/bin/env python3
"""
Model Status Checker
Check if models are trained and available, with data statistics
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import get_model_path, DATA_ROOT, PARQUET_DATA_DIR, MODELS_DIR
from src.data_collector.local_storage import list_parquet_files
from src.model_training.local_data_loader import fetch_parquet_data_from_local

def check_model_files():
    """Check if model files exist and get their info"""
    print("🤖 **MODEL AVAILABILITY STATUS**")
    print("=" * 50)
    
    # PRODUCTION FIX: Check for ensemble models (new production standard)
    models = {
        "Production Ensemble": {
            "model_file": MODELS_DIR / "production_ensemble_*.joblib",
            "scaler_file": MODELS_DIR / "production_scaler_*.joblib", 
            "features_file": MODELS_DIR / "production_features_*.json",
            "name": "production_ensemble"
        },
        "Random Forest": {
            "model_file": get_model_path("random_forest", "trained_model.pkl"),
            "features_file": get_model_path("random_forest", "expected_features.json"),
            "name": "random_forest"
        },
        "XGBoost": {
            "model_file": get_model_path("xgboost", "trained_model.pkl"),
            "features_file": get_model_path("xgboost", "expected_features.json"),
            "name": "xgboost"
        }
    }
    
    trained_models = []
    
    for model_name, paths in models.items():
        if model_name == "Production Ensemble":
            # Handle glob patterns for ensemble models
            import glob
            ensemble_files = list(Path(MODELS_DIR).glob("production_ensemble_*.joblib"))
            scaler_files = list(Path(MODELS_DIR).glob("production_scaler_*.joblib"))
            feature_files = list(Path(MODELS_DIR).glob("production_features_*.json"))
            
            if ensemble_files and scaler_files and feature_files:
                # Get the most recent files
                latest_ensemble = max(ensemble_files, key=lambda x: x.stat().st_mtime)
                latest_scaler = max(scaler_files, key=lambda x: x.stat().st_mtime)
                latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
                
                model_size = latest_ensemble.stat().st_size / (1024 * 1024)  # MB
                model_time = datetime.fromtimestamp(latest_ensemble.stat().st_mtime)
                
                print(f"✅ **{model_name}**: TRAINED & READY (PRODUCTION)")
                print(f"   📁 Ensemble Model: {latest_ensemble}")
                print(f"   🔧 Scaler: {latest_scaler}")
                print(f"   📋 Features: {latest_features}")
                print(f"   📏 Size: {model_size:.2f} MB")
                print(f"   🕐 Last Modified: {model_time}")
                
                # Try to read expected features
                try:
                    with open(latest_features, 'r') as f:
                        features = json.load(f)
                    print(f"   🎯 Features: {len(features)}")
                    print(f"   📊 Sample Features: {', '.join(features[:5])}...")
                except Exception as e:
                    print(f"   ⚠️ Could not read features: {e}")
                
                trained_models.append(paths["name"])
                
            else:
                print(f"❌ **{model_name}**: NOT TRAINED")
                print(f"   ❌ Missing ensemble files in {MODELS_DIR}")
        else:
            # Handle regular model files
            model_file = Path(paths["model_file"])
            features_file = Path(paths["features_file"])
            
            if model_file.exists() and features_file.exists():
                # Get file info
                model_size = model_file.stat().st_size / (1024 * 1024)  # MB
                model_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                
                print(f"✅ **{model_name}**: TRAINED & READY (LEGACY)")
                print(f"   📁 Model File: {model_file}")
                print(f"   📏 Size: {model_size:.2f} MB")
                print(f"   🕐 Last Modified: {model_time}")
                
                # Try to read expected features
                try:
                    with open(features_file, 'r') as f:
                        features = json.load(f)
                    print(f"   🎯 Expected Features: {len(features)}")
                    print(f"   📋 Sample Features: {', '.join(features[:5])}...")
                except Exception as e:
                    print(f"   ⚠️ Could not read features: {e}")
                
                trained_models.append(paths["name"])
                
            elif model_file.exists():
                print(f"⚠️ **{model_name}**: MODEL EXISTS (missing features file)")
                print(f"   📁 Model File: {model_file}")
                print(f"   ❌ Missing: {features_file}")
                
            else:
                print(f"❌ **{model_name}**: NOT TRAINED")
                print(f"   ❌ Missing: {model_file}")
                if not features_file.exists():
                    print(f"   ❌ Missing: {features_file}")
    
    print(f"\n📊 **SUMMARY**: {len(trained_models)} models trained")
    if trained_models:
        print(f"✅ Available Models: {', '.join(trained_models)}")
        if "production_ensemble" in trained_models:
            print("🎉 **PRODUCTION ENSEMBLE READY** - Best performance expected!")
    else:
        print("❌ No trained models found - run model training first!")
    
    return trained_models

def check_data_status():
    """Check local data availability and statistics"""
    print("\n📊 **LOCAL DATA STATUS**")
    print("=" * 50)
    
    # Check parquet files
    parquet_files = list_parquet_files()
    
    if not parquet_files:
        print("❌ No parquet files found in local storage")
        print(f"📁 Data Directory: {PARQUET_DATA_DIR}")
        return
    
    print(f"📁 Data Directory: {PARQUET_DATA_DIR}")
    print(f"📋 Found {len(parquet_files)} parquet files")
    
    # Try to load data for training analysis
    try:
        print("\n🔄 Loading data for analysis...")
        df = fetch_parquet_data_from_local()
        
        if df is not None and not df.empty:
            symbol_counts = df['symbol'].value_counts()
            total_rows = len(df)
            unique_symbols = df['symbol'].nunique()
            
            try:
                date_range = df['timestamp'].agg(['min', 'max'])
                print(f"📈 **TRAINING DATA READY:**")
                print(f"   📊 Total Rows: {total_rows:,}")
                print(f"   🎯 Unique Symbols: {unique_symbols}")
                print(f"   📅 Date Range: {date_range['min']} to {date_range['max']}")
                print(f"   📋 Top Symbols by Row Count:")
                for symbol, count in symbol_counts.head(10).items():
                    print(f"      • {symbol}: {count:,} rows")
                
                # Check if enough for training
                if total_rows >= 500:
                    print(f"✅ **Sufficient data for training** ({total_rows:,} >= 500 rows)")
                else:
                    print(f"⚠️ **Limited data for training** ({total_rows:,} < 500 rows)")
                    
            except Exception as e:
                print(f"   ⚠️ Error analyzing timestamps: {e}")
                print(f"   📊 Total Rows: {total_rows:,}")
                print(f"   🎯 Unique Symbols: {unique_symbols}")
        else:
            print("❌ No valid data loaded from local storage")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        
        # Show individual file info as fallback
        print("\n📁 **Individual File Info:**")
        for file_info in parquet_files[:10]:  # Show first 10 files
            size_kb = file_info['size'] / 1024
            print(f"   📄 {file_info['filename']}: {size_kb:.1f} KB")

def main():
    """Main status check function"""
    print("🔍 **MONEY PRINTER MODEL & DATA STATUS CHECK**")
    print("=" * 60)
    print(f"🕐 Check Time: {datetime.now()}")
    print("=" * 60)
    
    # Check model status
    trained_models = check_model_files()
    
    # Check data status
    check_data_status()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 **FINAL STATUS SUMMARY**")
    print("=" * 60)
    
    # PRODUCTION FIX: Updated recommendations for ensemble models
    if "production_ensemble" in trained_models:
        print("🎉 **PRODUCTION READY**: Ensemble model trained and available!")
        print("✅ Ready for: Live trading with validated ensemble model")
        print("💡 Recommended: Use /start_live_trade for production trading")
    elif len(trained_models) >= 2:
        print("⚠️ **LEGACY MODELS READY**: Multiple legacy models available")
        print("💡 Recommendation: Upgrade to ensemble model for better performance")
        print("   Run: python src/model_training/ensemble_production_trainer.py")
    elif len(trained_models) == 1:
        print("⚠️ **SINGLE MODEL READY**: One legacy model trained")
        print("💡 Recommendation: Train ensemble model for production use")
        print("   Run: python src/model_training/ensemble_production_trainer.py")
    else:
        print("❌ **NOT READY**: No trained models")
        print("💡 Next Steps for Production:")
        print("   1. Start data collection: python src/data_collector/production_data_scraper.py --hours 12")
        print("   2. Train ensemble model: python src/model_training/ensemble_production_trainer.py")
        print("   3. Validate system: python check_model_status.py")
        print("   4. Start live trading: python src/trading_bot/trade_runner.py")
    
    print(f"\n🔄 To run this check again: python check_model_status.py")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick Production Trainer - Simplified version for immediate deployment
Uses only basic RandomForest model for fast training
"""
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.storage.minimal_storage_manager import MinimalStorageManager
from src.config import DATA_ROOT, MODELS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quick_train():
    """Quick training function using minimal dependencies"""
    print("🚀 **QUICK PRODUCTION TRAINER**")
    print("="*50)
    
    # Initialize storage
    storage_manager = MinimalStorageManager(
        local_backup_dir=str(DATA_ROOT / "scraped_data" / "parquet_files")
    )
    
    # Load data
    print("📊 Loading data...")
    data = storage_manager.load_combined_data(min_rows=50)
    
    if data.empty:
        print("❌ No data loaded - cannot train")
        return False
        
    print(f"✅ Loaded {len(data)} rows")
    
    # Create features
    print("🔧 Creating features...")
    
    # Simple feature engineering
    features = []
    
    # Technical indicators
    if 'rsi' in data.columns:
        features.append('rsi')
    if 'macd' in data.columns:
        features.append('macd')
    if 'volume_change' in data.columns:
        features.append('volume_change')
    
    # Price features
    if 'close' in data.columns and 'open' in data.columns:
        data['price_change_pct'] = ((data['close'] - data['open']) / data['open']) * 100
        features.append('price_change_pct')
        
    # Volume features
    if 'volume' in data.columns:
        data['volume_zscore'] = (data['volume'] - data['volume'].mean()) / data['volume'].std()
        features.append('volume_zscore')
    
    # Use available numeric columns as backup
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    backup_features = [col for col in numeric_cols if col not in features and col != 'target']
    
    if len(features) < 3:
        features.extend(backup_features[:5])  # Take top 5 backup features
        
    if len(features) < 2:
        print("❌ Insufficient features for training")
        return False
        
    print(f"📋 Using features: {features}")
    
    # Create target
    if 'target' not in data.columns:
        # Create a simple target based on future price movement
        if 'close' in data.columns:
            data['future_return'] = data['close'].pct_change().shift(-1)
            data['target'] = (data['future_return'] > 0.01).astype(int)  # 1% threshold
        else:
            print("❌ Cannot create target - no price data")
            return False
    
    # Prepare training data
    X = data[features].fillna(0)
    y = data['target'].fillna(0)
    
    # Remove any infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 100:
        print(f"❌ Insufficient valid samples: {len(X)}")
        return False
        
    print(f"✅ Training samples: {len(X)}")
    
    # Train model
    print("🤖 Training RandomForest model...")
    
    # Use basic imports to avoid hangs
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("📊 **MODEL PERFORMANCE**")
    print(f"   🎯 Accuracy: {accuracy:.4f}")
    print(f"   🔧 Precision: {precision:.4f}")
    print(f"   📈 Recall: {recall:.4f}")
    print(f"   ⚖️ F1-Score: {f1:.4f}")
    
    # Save model
    model_dir = Path(MODELS_DIR) / "quick_production"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "trained_model.pkl"
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForest',
        'features': features,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'trained_at': datetime.now().isoformat(),
        'feature_count': len(features)
    }
    
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    expected_features_path = model_dir / "expected_features.json"
    with open(expected_features_path, 'w') as f:
        json.dump(features, f, indent=2)
    
    print("💾 **MODEL SAVED**")
    print(f"   📁 Model: {model_path}")
    print(f"   📋 Metadata: {metadata_path}")
    print(f"   🔧 Features: {expected_features_path}")
    
    return True

if __name__ == "__main__":
    success = quick_train()
    if success:
        print("\n🎉 **QUICK TRAINING COMPLETED SUCCESSFULLY!**")
    else:
        print("\n❌ **TRAINING FAILED**")
        sys.exit(1)

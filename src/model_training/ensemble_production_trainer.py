"""
Production Ensemble ML Trainer - Real model training with strict validation
Implements multiple models with proper ensemble stacking for production use
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import VotingClassifier

# Optional imports for advanced models
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.minimal_storage_manager import MinimalStorageManager as StorageManager
from src.config import MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

class EnsembleProductionTrainer:
    """Production-grade ensemble trainer with strict validation"""
    
    def __init__(self, 
                 storage_manager: StorageManager = None,
                 min_rows_total: int = 1000,
                 min_rows_per_symbol: int = 100,
                 min_time_span_hours: int = 12,
                 model_output_dir: str = None):
        
        self.storage_manager = storage_manager
        self.min_rows_total = min_rows_total
        self.min_rows_per_symbol = min_rows_per_symbol
        self.min_time_span_hours = min_time_span_hours
        self.model_output_dir = Path(model_output_dir or MODELS_DIR)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # PRODUCTION FIX: Initialize all required models
        self.models = {}
        self.ensemble_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        
        # Training data
        self.raw_data = None
        self.processed_data = None
        self.validation_results = {}
        
        logger.info(f"🎯 PRODUCTION ENSEMBLE TRAINER INITIALIZED")
        logger.info(f"   📊 Min total rows: {self.min_rows_total:,}")
        logger.info(f"   📈 Min rows per symbol: {self.min_rows_per_symbol}")
        logger.info(f"   ⏱️ Min time span: {self.min_time_span_hours}h")
        logger.info(f"   💾 Model output: {self.model_output_dir}")
    
    def validate_data_for_production(self, data: pd.DataFrame) -> Dict[str, Any]:
        """STRICT production data validation - no training on insufficient data"""
        validation = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(data),
            'is_production_ready': False,
            'critical_failures': [],
            'warnings': [],
            'passed_checks': [],
            'symbol_stats': {},
            'time_span_hours': 0,
            'data_quality_score': 0.0
        }
        
        logger.info("🔍 PRODUCTION DATA VALIDATION - STRICT MODE")
        
        # CRITICAL CHECK 1: Minimum rows (FATAL if failed)
        if len(data) < self.min_rows_total:
            validation['critical_failures'].append(f"FATAL: Insufficient rows {len(data):,} < {self.min_rows_total:,}")
            logger.error(f"❌ CRITICAL: Insufficient training data - {len(data):,} rows < {self.min_rows_total:,}")
            validation['is_production_ready'] = False
            return validation
        else:
            validation['passed_checks'].append(f"Total rows: {len(data):,} >= {self.min_rows_total:,}")
        
        # CRITICAL CHECK 2: Required columns (FATAL if failed)
        required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            validation['critical_failures'].append(f"FATAL: Missing critical columns: {missing_cols}")
            logger.error(f"❌ CRITICAL: Missing required columns: {missing_cols}")
            validation['is_production_ready'] = False
            return validation
        else:
            validation['passed_checks'].append("All required columns present")
        
        # CRITICAL CHECK 3: Time span (FATAL if failed)
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            time_span = data['timestamp'].max() - data['timestamp'].min()
            validation['time_span_hours'] = time_span.total_seconds() / 3600
            
            if validation['time_span_hours'] < self.min_time_span_hours:
                validation['critical_failures'].append(f"FATAL: Time span {validation['time_span_hours']:.1f}h < {self.min_time_span_hours}h")
                logger.error(f"❌ CRITICAL: Insufficient time span - {validation['time_span_hours']:.1f}h < {self.min_time_span_hours}h")
                validation['is_production_ready'] = False
                return validation
            else:
                validation['passed_checks'].append(f"Time span: {validation['time_span_hours']:.1f}h >= {self.min_time_span_hours}h")
        except Exception as e:
            validation['critical_failures'].append(f"FATAL: Time span calculation failed: {e}")
            logger.error(f"❌ CRITICAL: Time span validation failed: {e}")
            validation['is_production_ready'] = False
            return validation
        
        # CRITICAL CHECK 4: Symbol diversity (FATAL if failed)
        if 'symbol' in data.columns:
            symbol_counts = data['symbol'].value_counts()
            validation['symbol_stats'] = symbol_counts.to_dict()
            
            sufficient_symbols = sum(1 for count in symbol_counts.values() if count >= self.min_rows_per_symbol)
            if sufficient_symbols < 3:
                validation['critical_failures'].append(f"FATAL: Only {sufficient_symbols} symbols with sufficient data (need ≥3)")
                logger.error(f"❌ CRITICAL: Insufficient symbol diversity - only {sufficient_symbols} symbols with ≥{self.min_rows_per_symbol} rows")
                validation['is_production_ready'] = False
                return validation
            else:
                validation['passed_checks'].append(f"Sufficient symbols: {sufficient_symbols} ≥ 3")
        
        # CRITICAL CHECK 5: Data quality (FATAL if too poor)
        null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
        duplicate_percentage = (data.duplicated().sum() / len(data) * 100)
        
        if null_percentage > 20:
            validation['critical_failures'].append(f"FATAL: Excessive null values: {null_percentage:.1f}%")
            logger.error(f"❌ CRITICAL: Too many null values - {null_percentage:.1f}% > 20%")
            validation['is_production_ready'] = False
            return validation
        
        if duplicate_percentage > 50:
            validation['critical_failures'].append(f"FATAL: Excessive duplicates: {duplicate_percentage:.1f}%")
            logger.error(f"❌ CRITICAL: Too many duplicates - {duplicate_percentage:.1f}% > 50%")
            validation['is_production_ready'] = False
            return validation
        
        validation['data_quality_score'] = max(0, 100 - null_percentage - duplicate_percentage)
        validation['passed_checks'].append(f"Data quality: {validation['data_quality_score']:.1f}/100")
        
        # CRITICAL CHECK 6: Price data validity (FATAL if invalid)
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            price_cols = ['open', 'high', 'low', 'close']
            price_data = data[price_cols].apply(pd.to_numeric, errors='coerce')
            
            invalid_prices = (price_data <= 0).sum().sum()
            if invalid_prices > len(data) * 0.05:  # >5% invalid prices is fatal
                validation['critical_failures'].append(f"FATAL: Too many invalid prices: {invalid_prices}")
                logger.error(f"❌ CRITICAL: Too many invalid prices - {invalid_prices} > 5% of data")
                validation['is_production_ready'] = False
                return validation
            
            validation['passed_checks'].append("Price data validity confirmed")
        
        # If we made it here, data is production-ready
        validation['is_production_ready'] = True
        logger.info("✅ PRODUCTION VALIDATION PASSED - Data ready for training")
        
        return validation
    
    def load_production_data(self) -> Optional[pd.DataFrame]:
        """Load ALL available production data - no sampling, no test subsets"""
        if not self.storage_manager:
            logger.error("❌ FATAL: No storage manager configured")
            return None
        
        logger.info("📂 LOADING ALL PRODUCTION DATA - NO SAMPLING")
        
        # Get all data files
        files = self.storage_manager.list_files()
        data_files = [f for f in files if f.endswith('.parquet')]
        
        if not data_files:
            logger.error("❌ FATAL: No data files found in storage")
            return None
        
        logger.info(f"📋 Found {len(data_files)} production data files")
        
        # Load ALL data - no filtering, no sampling
        all_data = []
        loaded_files = 0
        total_rows = 0
        
        for filename in data_files:
            try:
                df = self.storage_manager.load_data(filename)
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    loaded_files += 1
                    total_rows += len(df)
                    logger.debug(f"📊 Loaded {filename}: {len(df)} rows")
                else:
                    logger.warning(f"⚠️ Skipped empty file: {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to load {filename}: {e}")
        
        if not all_data:
            logger.error("❌ FATAL: No valid data loaded from any file")
            return None
        
        # Combine ALL data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates (keep all unique data)
        initial_rows = len(combined_data)
        combined_data = combined_data.drop_duplicates()
        final_rows = len(combined_data)
        
        logger.info(f"📊 PRODUCTION DATA LOADED:")
        logger.info(f"   📁 Files processed: {loaded_files}/{len(data_files)}")
        logger.info(f"   📈 Total rows: {final_rows:,}")
        logger.info(f"   🧹 Duplicates removed: {initial_rows - final_rows:,}")
        
        if final_rows < self.min_rows_total:
            logger.error(f"❌ FATAL: Combined data insufficient - {final_rows:,} < {self.min_rows_total:,}")
            return None
        
        return combined_data
    
    def create_production_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for production trading"""
        logger.info("⚙️ CREATING PRODUCTION FEATURES")
        
        df = data.copy()
        
        # Ensure proper timestamp handling
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(['symbol', 'timestamp'])
        
        # PRODUCTION FIX: Comprehensive feature engineering
        feature_count = 0
        
        # Basic price features
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['high_low_spread'] = df['high'] - df['low']
        df['high_low_spread_pct'] = (df['high'] - df['low']) / df['low'] * 100
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        feature_count += 7
        
        # Technical indicators (multiple timeframes)
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df.groupby('symbol')['close'].rolling(window).mean().reset_index(0, drop=True)
            df[f'ema_{window}'] = df.groupby('symbol')['close'].ewm(span=window).mean().reset_index(0, drop=True)
            df[f'bb_upper_{window}'] = df[f'sma_{window}'] + 2 * df.groupby('symbol')['close'].rolling(window).std().reset_index(0, drop=True)
            df[f'bb_lower_{window}'] = df[f'sma_{window}'] - 2 * df.groupby('symbol')['close'].rolling(window).std().reset_index(0, drop=True)
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            feature_count += 5
        
        # Momentum indicators
        for period in [5, 10, 14, 20]:
            df[f'momentum_{period}'] = df.groupby('symbol')['close'].pct_change(period).reset_index(0, drop=True) * 100
            df[f'roc_{period}'] = ((df['close'] - df.groupby('symbol')['close'].shift(period)) / df.groupby('symbol')['close'].shift(period) * 100).reset_index(0, drop=True)
            feature_count += 2
        
        # Volatility features
        for window in [10, 20, 30]:
            df[f'volatility_{window}'] = df.groupby('symbol')['close'].rolling(window).std().reset_index(0, drop=True)
            df[f'atr_{window}'] = df.groupby('symbol').apply(
                lambda x: pd.Series(
                    np.maximum(
                        np.maximum(x['high'] - x['low'], abs(x['high'] - x['close'].shift())),
                        abs(x['low'] - x['close'].shift())
                    ).rolling(window).mean().values,
                    index=x.index
                )
            ).reset_index(0, drop=True)
            feature_count += 2
        
        # Volume features
        df['volume_ma_10'] = df.groupby('symbol')['volume'].rolling(10).mean().reset_index(0, drop=True)
        df['volume_ma_20'] = df.groupby('symbol')['volume'].rolling(20).mean().reset_index(0, drop=True)
        df['volume_ratio'] = df['volume'] / df['volume_ma_10']
        df['volume_change'] = df.groupby('symbol')['volume'].pct_change().reset_index(0, drop=True)
        feature_count += 4
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['minute'] = df['timestamp'].dt.minute
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
            feature_count += 5
        
        # PRODUCTION TARGET: Next period direction (buy/sell/hold)
        df['next_return'] = df.groupby('symbol')['close'].pct_change(-1).reset_index(0, drop=True) * 100
        
        # Create classification target (not regression)
        df['target'] = 0  # Hold
        df.loc[df['next_return'] > 1.0, 'target'] = 1  # Buy (>1% gain expected)
        df.loc[df['next_return'] < -1.0, 'target'] = 2  # Sell (<-1% loss expected)
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target'])
        
        logger.info(f"⚙️ FEATURES CREATED: {feature_count} features, {len(df):,} samples")
        logger.info(f"   🎯 Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def train_ensemble_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train full ensemble of production models"""
        logger.info("🤖 TRAINING PRODUCTION ENSEMBLE MODELS")
        
        training_results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'training_samples': 0,
            'test_samples': 0,
            'features_used': 0,
            'individual_models': {},
            'ensemble_performance': {},
            'model_paths': {},
            'feature_importance': {}
        }
        
        try:
            # Feature selection (exclude non-feature columns)
            exclude_cols = [
                'timestamp', 'symbol', 'collection_time', 'close_time', 
                'target', 'next_return', 'open', 'high', 'low', 'close', 'volume',
                'interval'
            ]
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            # Remove high-NaN features
            feature_cols = [col for col in feature_cols if data[col].notna().sum() > len(data) * 0.7]
            
            if len(feature_cols) < 10:
                raise ValueError(f"FATAL: Insufficient features for production - only {len(feature_cols)} available")
            
            self.feature_columns = feature_cols
            logger.info(f"📊 PRODUCTION FEATURES: {len(feature_cols)} selected")
            
            # Prepare data
            X = data[feature_cols].fillna(0)
            y = data['target']
            
            # Clean infinite values
            X = X.replace([np.inf, -np.inf], 0)
            
            # Stratified split for classification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=RANDOM_STATE, 
                stratify=y, shuffle=True
            )
            
            training_results['training_samples'] = len(X_train)
            training_results['test_samples'] = len(X_test)
            training_results['features_used'] = len(feature_cols)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # PRODUCTION MODELS (conditional based on available packages)
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ),
                'logistic_regression': LogisticRegression(
                    class_weight='balanced',
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    solver='liblinear'
                )
            }
            
            # Add XGBoost if available
            if HAS_XGBOOST:
                models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
                logger.info("✅ XGBoost added to ensemble")
            else:
                logger.warning("⚠️ XGBoost not available - using RF + LR ensemble")
            
            # Add LightGBM if available
            if HAS_LIGHTGBM:
                models['lightgbm'] = lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    class_weight='balanced'
                )
                logger.info("✅ LightGBM added to ensemble")
            else:
                logger.warning("⚠️ LightGBM not available - using available models")
            
            # Train individual models
            trained_models = []
            for name, model in models.items():
                logger.info(f"🔄 Training {name}...")
                
                try:
                    model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                    
                    # Metrics
                    metrics = {
                        'train_accuracy': float(accuracy_score(y_train, train_pred)),
                        'test_accuracy': float(accuracy_score(y_test, test_pred)),
                        'train_f1': float(f1_score(y_train, train_pred, average='weighted')),
                        'test_f1': float(f1_score(y_test, test_pred, average='weighted')),
                        'precision': float(precision_score(y_test, test_pred, average='weighted')),
                        'recall': float(recall_score(y_test, test_pred, average='weighted'))
                    }
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
                    metrics['cv_f1_mean'] = float(cv_scores.mean())
                    metrics['cv_f1_std'] = float(cv_scores.std())
                    
                    training_results['individual_models'][name] = metrics
                    trained_models.append((name, model))
                    
                    logger.info(f"✅ {name}: Test F1={metrics['test_f1']:.4f}, Accuracy={metrics['test_accuracy']:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to train {name}: {e}")
            
            if len(trained_models) < 2:
                raise ValueError("FATAL: Not enough models trained for ensemble")
            
            # Create ensemble
            ensemble_models = [(name, model) for name, model in trained_models]
            self.ensemble_model = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'
            )
            
            logger.info("🎯 Training ensemble model...")
            self.ensemble_model.fit(X_train_scaled, y_train)
            
            # Ensemble evaluation
            ensemble_train_pred = self.ensemble_model.predict(X_train_scaled)
            ensemble_test_pred = self.ensemble_model.predict(X_test_scaled)
            
            training_results['ensemble_performance'] = {
                'train_accuracy': float(accuracy_score(y_train, ensemble_train_pred)),
                'test_accuracy': float(accuracy_score(y_test, ensemble_test_pred)),
                'train_f1': float(f1_score(y_train, ensemble_train_pred, average='weighted')),
                'test_f1': float(f1_score(y_test, ensemble_test_pred, average='weighted')),
                'precision': float(precision_score(y_test, ensemble_test_pred, average='weighted')),
                'recall': float(recall_score(y_test, ensemble_test_pred, average='weighted')),
                'confusion_matrix': confusion_matrix(y_test, ensemble_test_pred).tolist()
            }
            
            # Save models
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save ensemble
            ensemble_path = self.model_output_dir / f"production_ensemble_{timestamp_str}.joblib"
            joblib.dump(self.ensemble_model, ensemble_path)
            training_results['model_paths']['ensemble'] = str(ensemble_path)
            
            # Save scaler
            scaler_path = self.model_output_dir / f"production_scaler_{timestamp_str}.joblib"
            joblib.dump(self.scaler, scaler_path)
            training_results['model_paths']['scaler'] = str(scaler_path)
            
            # Save features
            features_path = self.model_output_dir / f"production_features_{timestamp_str}.json"
            with open(features_path, 'w') as f:
                json.dump(feature_cols, f)
            training_results['model_paths']['features'] = str(features_path)
            
            training_results['success'] = True
            
            # PRODUCTION LOGGING
            logger.info("🎉 PRODUCTION ENSEMBLE TRAINING COMPLETED!")
            logger.info(f"   📊 Training samples: {training_results['training_samples']:,}")
            logger.info(f"   🧪 Test samples: {training_results['test_samples']:,}")
            logger.info(f"   ⚙️ Features: {training_results['features_used']}")
            logger.info(f"   🎯 Ensemble Test Accuracy: {training_results['ensemble_performance']['test_accuracy']:.4f}")
            logger.info(f"   📈 Ensemble Test F1: {training_results['ensemble_performance']['test_f1']:.4f}")
            logger.info(f"   🔧 Ensemble Test Precision: {training_results['ensemble_performance']['precision']:.4f}")
            logger.info(f"   📊 Ensemble Test Recall: {training_results['ensemble_performance']['recall']:.4f}")
            logger.info(f"   💾 Models saved to: {self.model_output_dir}")
            
            return training_results
            
        except Exception as e:
            error_msg = f"FATAL: Ensemble training failed: {e}"
            logger.error(f"❌ {error_msg}")
            training_results['error'] = error_msg
            return training_results
    
    def run_production_training_pipeline(self) -> Dict[str, Any]:
        """Run complete production training pipeline with strict validation"""
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_success': False,
            'data_validation': None,
            'training_results': None,
            'error': None
        }
        
        logger.info("🚀 STARTING PRODUCTION ENSEMBLE TRAINING PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load ALL production data
            logger.info("📂 STEP 1: Loading ALL production data...")
            raw_data = self.load_production_data()
            
            if raw_data is None:
                raise ValueError("FATAL: No production data loaded")
            
            self.raw_data = raw_data
            
            # Step 2: STRICT validation
            logger.info("🔍 STEP 2: STRICT production validation...")
            validation_results = self.validate_data_for_production(raw_data)
            pipeline_results['data_validation'] = validation_results
            
            if not validation_results['is_production_ready']:
                logger.error("❌ PRODUCTION VALIDATION FAILED - ABORTING TRAINING")
                for failure in validation_results['critical_failures']:
                    logger.error(f"   🚨 {failure}")
                raise ValueError("Data validation failed - cannot proceed with production training")
            
            logger.info("✅ PRODUCTION VALIDATION PASSED")
            
            # Step 3: Create production features
            logger.info("⚙️ STEP 3: Creating production features...")
            processed_data = self.create_production_features(raw_data)
            
            if processed_data is None or len(processed_data) == 0:
                raise ValueError("FATAL: Feature creation failed")
            
            self.processed_data = processed_data
            
            # Step 4: Train ensemble
            logger.info("🤖 STEP 4: Training production ensemble...")
            training_results = self.train_ensemble_models(processed_data)
            pipeline_results['training_results'] = training_results
            
            if not training_results['success']:
                raise ValueError(f"FATAL: Ensemble training failed: {training_results.get('error', 'Unknown error')}")
            
            pipeline_results['pipeline_success'] = True
            logger.info("🎉 PRODUCTION TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            error_msg = f"FATAL: Production training pipeline failed: {e}"
            logger.error(f"❌ {error_msg}")
            pipeline_results['error'] = error_msg
        
        return pipeline_results

def main():
    """Standalone execution for production training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Ensemble Trainer')
    parser.add_argument('--min-rows', type=int, default=2000,
                       help='Minimum total rows required (default: 2000)')
    parser.add_argument('--min-hours', type=int, default=24,
                       help='Minimum time span in hours (default: 24)')
    parser.add_argument('--model-dir', type=str,
                       help='Model output directory')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('production_ensemble_trainer.log')
        ]
    )
    
    # Create storage manager (implement based on your setup)
    from src.config import DATA_ROOT
    storage_manager = StorageManager(
        local_backup_dir=str(DATA_ROOT / "scraped_data" / "parquet_files")
    )
    
    # Create trainer
    trainer = EnsembleProductionTrainer(
        storage_manager=storage_manager,
        min_rows_total=args.min_rows,
        min_time_span_hours=args.min_hours,
        model_output_dir=args.model_dir
    )
    
    # Run training
    results = trainer.run_production_training_pipeline()
    
    # Print results
    print("\n" + "="*60)
    print("PRODUCTION ENSEMBLE TRAINING RESULTS")
    print("="*60)
    
    if results['pipeline_success']:
        training = results['training_results']
        ensemble_perf = training['ensemble_performance']
        print(f"🎉 TRAINING SUCCESSFUL!")
        print(f"📊 Training samples: {training['training_samples']:,}")
        print(f"🧪 Test samples: {training['test_samples']:,}")
        print(f"⚙️ Features used: {training['features_used']}")
        print(f"🎯 Ensemble Accuracy: {ensemble_perf['test_accuracy']:.4f}")
        print(f"📈 Ensemble F1-Score: {ensemble_perf['test_f1']:.4f}")
        print(f"🔧 Ensemble Precision: {ensemble_perf['precision']:.4f}")
        print(f"📊 Ensemble Recall: {ensemble_perf['recall']:.4f}")
        print(f"💾 Models saved: {len(training['model_paths'])} files")
    else:
        print(f"❌ TRAINING FAILED: {results.get('error', 'Unknown error')}")
        
        if results.get('data_validation'):
            validation = results['data_validation']
            if validation.get('critical_failures'):
                print("\n🚨 CRITICAL FAILURES:")
                for failure in validation['critical_failures']:
                    print(f"   • {failure}")

if __name__ == "__main__":
    main()

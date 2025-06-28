"""
Production Machine Learning Trainer - Robust model training with strict data validation
Only trains on sufficient, real data with comprehensive validation
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.enhanced_storage_manager import EnhancedStorageManager
from src.config import Config

logger = logging.getLogger(__name__)

class ProductionModelTrainer:
    """Production-ready ML trainer with strict data validation"""
    
    def __init__(self, 
                 storage_manager: EnhancedStorageManager = None,
                 min_rows_total: int = 500,
                 min_rows_per_symbol: int = 50,
                 min_time_span_hours: int = 6,
                 model_output_dir: str = None):
        
        self.storage_manager = storage_manager
        self.min_rows_total = min_rows_total
        self.min_rows_per_symbol = min_rows_per_symbol
        self.min_time_span_hours = min_time_span_hours
        self.model_output_dir = model_output_dir or "models"
        
        # Ensure model directory exists
        os.makedirs(self.model_output_dir, exist_ok=True)
        
        # Training state
        self.raw_data = None
        self.processed_data = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.validation_results = {}
        
        logger.info(f"🤖 Production trainer initialized")
        logger.info(f"   📊 Min total rows: {self.min_rows_total:,}")
        logger.info(f"   📈 Min rows per symbol: {self.min_rows_per_symbol}")
        logger.info(f"   ⏱️ Min time span: {self.min_time_span_hours}h")
        logger.info(f"   💾 Model output: {self.model_output_dir}")
    
    def validate_data_sufficiency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data validation before training"""
        validation = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(data),
            'is_sufficient': False,
            'issues': [],
            'passed_checks': [],
            'symbol_stats': {},
            'time_span_hours': 0,
            'data_quality_score': 0.0
        }
        
        # Check 1: Minimum total rows
        if len(data) >= self.min_rows_total:
            validation['passed_checks'].append(f"Total rows: {len(data):,} >= {self.min_rows_total:,}")
        else:
            validation['issues'].append(f"Insufficient total rows: {len(data):,} < {self.min_rows_total:,}")
        
        # Check 2: Required columns
        required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if not missing_cols:
            validation['passed_checks'].append("All required columns present")
        else:
            validation['issues'].append(f"Missing columns: {missing_cols}")
        
        # Check 3: Time span validation
        if 'timestamp' in data.columns:
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                time_span = data['timestamp'].max() - data['timestamp'].min()
                validation['time_span_hours'] = time_span.total_seconds() / 3600
                
                if validation['time_span_hours'] >= self.min_time_span_hours:
                    validation['passed_checks'].append(f"Time span: {validation['time_span_hours']:.1f}h >= {self.min_time_span_hours}h")
                else:
                    validation['issues'].append(f"Insufficient time span: {validation['time_span_hours']:.1f}h < {self.min_time_span_hours}h")
            except Exception as e:
                validation['issues'].append(f"Time span calculation failed: {e}")
        
        # Check 4: Per-symbol data sufficiency
        if 'symbol' in data.columns:
            symbol_counts = data['symbol'].value_counts()
            validation['symbol_stats'] = symbol_counts.to_dict()
            
            sufficient_symbols = 0
            for symbol, count in symbol_counts.items():
                if count >= self.min_rows_per_symbol:
                    sufficient_symbols += 1
                    
            if sufficient_symbols >= 3:  # At least 3 symbols with sufficient data
                validation['passed_checks'].append(f"Symbols with sufficient data: {sufficient_symbols}")
            else:
                validation['issues'].append(f"Too few symbols with sufficient data: {sufficient_symbols} < 3")
        
        # Check 5: Data quality - null values, duplicates, etc.
        null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
        duplicate_percentage = (data.duplicated().sum() / len(data) * 100)
        
        quality_issues = []
        if null_percentage > 5:
            quality_issues.append(f"High null values: {null_percentage:.1f}%")
        if duplicate_percentage > 10:
            quality_issues.append(f"High duplicates: {duplicate_percentage:.1f}%")
        
        if not quality_issues:
            validation['passed_checks'].append(f"Good data quality: {null_percentage:.1f}% nulls, {duplicate_percentage:.1f}% duplicates")
            validation['data_quality_score'] = max(0, 100 - null_percentage - duplicate_percentage)
        else:
            validation['issues'].extend(quality_issues)
            validation['data_quality_score'] = max(0, 50 - null_percentage - duplicate_percentage)
        
        # Check 6: Price data validity
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            price_cols = ['open', 'high', 'low', 'close']
            price_data = data[price_cols].apply(pd.to_numeric, errors='coerce')
            
            # Check for reasonable price ranges
            invalid_prices = (price_data <= 0).sum().sum()
            if invalid_prices == 0:
                validation['passed_checks'].append("All prices are positive")
            else:
                validation['issues'].append(f"Invalid prices found: {invalid_prices}")
            
            # Check OHLC logic (High >= Low, etc.)
            ohlc_violations = 0
            if len(price_data) > 0:
                ohlc_violations += (price_data['high'] < price_data['low']).sum()
                ohlc_violations += (price_data['high'] < price_data['open']).sum()
                ohlc_violations += (price_data['high'] < price_data['close']).sum()
                ohlc_violations += (price_data['low'] > price_data['open']).sum()
                ohlc_violations += (price_data['low'] > price_data['close']).sum()
            
            if ohlc_violations == 0:
                validation['passed_checks'].append("OHLC data is logically consistent")
            else:
                validation['issues'].append(f"OHLC logic violations: {ohlc_violations}")
        
        # Overall sufficiency determination
        critical_checks = [
            len(data) >= self.min_rows_total,
            validation['time_span_hours'] >= self.min_time_span_hours,
            not missing_cols,
            validation['data_quality_score'] > 30
        ]
        
        validation['is_sufficient'] = all(critical_checks)
        validation['passed_checks_count'] = len(validation['passed_checks'])
        validation['issues_count'] = len(validation['issues'])
        
        return validation
    
    def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load and combine all available training data"""
        if not self.storage_manager:
            logger.error("❌ No storage manager configured")
            return None
        
        logger.info("📂 Loading training data from storage...")
        
        # Get all available data files
        files = self.storage_manager.list_files()
        data_files = [f for f in files if f.endswith('.parquet')]
        
        if not data_files:
            logger.warning("⚠️ No data files found in storage")
            return None
        
        logger.info(f"📋 Found {len(data_files)} data files")
        
        # Load and combine all data
        all_data = []
        loaded_files = 0
        
        for filename in data_files:
            try:
                df = self.storage_manager.load_data(filename)
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    loaded_files += 1
                    logger.debug(f"📊 Loaded {filename}: {len(df)} rows")
                else:
                    logger.warning(f"⚠️ Empty or invalid file: {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to load {filename}: {e}")
        
        if not all_data:
            logger.error("❌ No valid data loaded")
            return None
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        initial_rows = len(combined_data)
        combined_data = combined_data.drop_duplicates()
        final_rows = len(combined_data)
        
        logger.info(f"📊 Combined data: {final_rows:,} rows from {loaded_files} files")
        if initial_rows != final_rows:
            logger.info(f"🧹 Removed {initial_rows - final_rows:,} duplicate rows")
        
        return combined_data
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from raw market data"""
        logger.info("⚙️ Creating features from market data...")
        
        df = data.copy()
        
        # Ensure datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Encode categorical features
        if 'interval' in df.columns:
            interval_mapping = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '1d': 1440}
            df['interval_minutes'] = df['interval'].map(interval_mapping).fillna(60)
        
        # Basic price features
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['high_low_spread'] = df['high'] - df['low']
        df['high_low_spread_pct'] = (df['high'] - df['low']) / df['low'] * 100
        
        # Technical indicators
        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df.groupby('symbol')['close'].rolling(window).mean().reset_index(0, drop=True)
            df[f'ema_{window}'] = df.groupby('symbol')['close'].ewm(span=window).mean().reset_index(0, drop=True)
            df[f'volatility_{window}'] = df.groupby('symbol')['close'].rolling(window).std().reset_index(0, drop=True)
        
        # RSI-like momentum indicator
        df['momentum_5'] = df.groupby('symbol')['close'].pct_change(5).reset_index(0, drop=True)
        df['momentum_10'] = df.groupby('symbol')['close'].pct_change(10).reset_index(0, drop=True)
        
        # Volume features
        df['volume_ma_10'] = df.groupby('symbol')['volume'].rolling(10).mean().reset_index(0, drop=True)
        df['volume_ratio'] = df['volume'] / df['volume_ma_10']
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['minute'] = df['timestamp'].dt.minute
        
        # Target variable - next period return
        df['target'] = df.groupby('symbol')['close'].pct_change(-1).reset_index(0, drop=True) * 100  # Next return in %
        
        # Remove rows with NaN targets (last row of each symbol)
        df = df.dropna(subset=['target'])
        
        logger.info(f"⚙️ Created {len(df.columns)} features, {len(df):,} training samples")
        return df
    
    def train_model(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the machine learning model"""
        logger.info("🤖 Training machine learning model...")
        training_results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'model_type': 'RandomForestRegressor',
            'training_samples': 0,
            'test_samples': 0,
            'features_used': 0,
            'performance': {},
            'model_path': None,
            'scaler_path': None
        }
        
        try:
            # Select features (exclude non-feature columns)
            exclude_cols = ['timestamp', 'symbol', 'collection_time', 'close_time', 
                           'target', 'open', 'high', 'low', 'close', 'interval']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            # Remove columns with too many NaN values
            feature_cols = [col for col in feature_cols if data[col].notna().sum() > len(data) * 0.8]
            
            if len(feature_cols) < 5:
                raise ValueError(f"Insufficient features: only {len(feature_cols)} available")
            
            self.feature_columns = feature_cols
            logger.info(f"📊 Using {len(feature_cols)} features: {feature_cols[:10]}...")
            
            # Prepare data
            X = data[feature_cols].fillna(0)
            y = data['target'].fillna(0)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )
            
            training_results['training_samples'] = len(X_train)
            training_results['test_samples'] = len(X_test)
            training_results['features_used'] = len(feature_cols)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("🌲 Training Random Forest model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            training_results['performance'] = {
                'train_mse': float(mean_squared_error(y_train, train_pred)),
                'test_mse': float(mean_squared_error(y_test, test_pred)),
                'train_mae': float(mean_absolute_error(y_train, train_pred)),
                'test_mae': float(mean_absolute_error(y_test, test_pred)),
                'train_r2': float(r2_score(y_train, train_pred)),
                'test_r2': float(r2_score(y_test, test_pred))
            }
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
            training_results['performance']['cv_r2_mean'] = float(cv_scores.mean())
            training_results['performance']['cv_r2_std'] = float(cv_scores.std())
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            training_results['top_features'] = [(name, float(importance)) for name, importance in top_features]
            
            # Save model and scaler
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"production_model_{timestamp_str}.joblib"
            scaler_filename = f"production_scaler_{timestamp_str}.joblib"
            
            model_path = os.path.join(self.model_output_dir, model_filename)
            scaler_path = os.path.join(self.model_output_dir, scaler_filename)
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature list
            features_path = os.path.join(self.model_output_dir, f"production_features_{timestamp_str}.json")
            with open(features_path, 'w') as f:
                json.dump(feature_cols, f)
            
            training_results['model_path'] = model_path
            training_results['scaler_path'] = scaler_path
            training_results['features_path'] = features_path
            training_results['success'] = True
            
            # Log results
            perf = training_results['performance']
            logger.info("✅ Model training completed successfully!")
            logger.info(f"   📊 Training samples: {training_results['training_samples']:,}")
            logger.info(f"   🧪 Test samples: {training_results['test_samples']:,}")
            logger.info(f"   ⚙️ Features: {training_results['features_used']}")
            logger.info(f"   📈 Test R²: {perf['test_r2']:.4f}")
            logger.info(f"   📉 Test MAE: {perf['test_mae']:.4f}")
            logger.info(f"   🔀 CV R²: {perf['cv_r2_mean']:.4f} ± {perf['cv_r2_std']:.4f}")
            logger.info(f"   💾 Model saved: {model_path}")
            
            return training_results
            
        except Exception as e:
            error_msg = f"Model training failed: {e}"
            logger.error(f"❌ {error_msg}")
            training_results['error'] = error_msg
            return training_results
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline with validation"""
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_success': False,
            'data_validation': None,
            'training_results': None,
            'error': None
        }
        
        logger.info("🚀 Starting full production training pipeline...")
        
        try:
            # Step 1: Load data
            logger.info("📂 Step 1: Loading training data...")
            raw_data = self.load_training_data()
            
            if raw_data is None:
                raise ValueError("No training data could be loaded")
            
            self.raw_data = raw_data
            
            # Step 2: Validate data sufficiency
            logger.info("✅ Step 2: Validating data sufficiency...")
            validation_results = self.validate_data_sufficiency(raw_data)
            pipeline_results['data_validation'] = validation_results
            
            # Log validation results
            logger.info(f"📊 Data validation results:")
            logger.info(f"   📈 Total rows: {validation_results['total_rows']:,}")
            logger.info(f"   ⏱️ Time span: {validation_results['time_span_hours']:.1f} hours")
            logger.info(f"   ✅ Passed checks: {validation_results['passed_checks_count']}")
            logger.info(f"   ❌ Issues: {validation_results['issues_count']}")
            logger.info(f"   🏆 Quality score: {validation_results['data_quality_score']:.1f}/100")
            
            if not validation_results['is_sufficient']:
                logger.error("❌ Data validation failed - insufficient data for training")
                logger.error("Issues found:")
                for issue in validation_results['issues']:
                    logger.error(f"   • {issue}")
                
                raise ValueError("Data validation failed - training requirements not met")
            
            logger.info("✅ Data validation passed - proceeding with training")
            
            # Step 3: Create features
            logger.info("⚙️ Step 3: Creating features...")
            processed_data = self.create_features(raw_data)
            
            if processed_data is None or len(processed_data) == 0:
                raise ValueError("Feature creation failed")
            
            self.processed_data = processed_data
            
            # Step 4: Train model
            logger.info("🤖 Step 4: Training model...")
            training_results = self.train_model(processed_data)
            pipeline_results['training_results'] = training_results
            
            if not training_results['success']:
                raise ValueError(f"Model training failed: {training_results.get('error', 'Unknown error')}")
            
            # Pipeline success
            pipeline_results['pipeline_success'] = True
            logger.info("🎉 Production training pipeline completed successfully!")
            
        except Exception as e:
            error_msg = f"Training pipeline failed: {e}"
            logger.error(f"❌ {error_msg}")
            pipeline_results['error'] = error_msg
        
        return pipeline_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if not self.model:
            return {'error': 'No model trained'}
        
        return {
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'feature_columns': self.feature_columns,
            'model_params': self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Model Trainer')
    parser.add_argument('--min-rows', type=int, default=1000,
                       help='Minimum total rows required (default: 1000)')
    parser.add_argument('--min-rows-per-symbol', type=int, default=100,
                       help='Minimum rows per symbol (default: 100)')
    parser.add_argument('--min-hours', type=int, default=24,
                       help='Minimum time span in hours (default: 24)')
    parser.add_argument('--memory-only', action='store_true',
                       help='Use memory-only storage (for testing)')
    parser.add_argument('--local-backup', type=str,
                       help='Local backup directory')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Model output directory')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('production_trainer.log')
        ]
    )
    
    # Load config
    from src.config import GOOGLE_DRIVE_FOLDER_ID
    
    # Create storage manager
    storage_manager = EnhancedStorageManager(
        drive_folder_id=GOOGLE_DRIVE_FOLDER_ID,
        local_backup_dir=args.local_backup,
        memory_only=args.memory_only
    )
    
    # Create trainer
    trainer = ProductionModelTrainer(
        storage_manager=storage_manager,
        min_rows_total=args.min_rows,
        min_rows_per_symbol=args.min_rows_per_symbol,
        min_time_span_hours=args.min_hours,
        model_output_dir=args.model_dir
    )
    
    # Run training
    results = trainer.run_full_training_pipeline()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING PIPELINE RESULTS")
    print("="*60)
    
    if results['pipeline_success']:
        training = results['training_results']
        print(f"✅ Training successful!")
        print(f"📊 Training samples: {training['training_samples']:,}")
        print(f"🧪 Test samples: {training['test_samples']:,}")
        print(f"⚙️ Features used: {training['features_used']}")
        print(f"📈 Test R²: {training['performance']['test_r2']:.4f}")
        print(f"💾 Model saved: {training['model_path']}")
    else:
        print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
        
        if results.get('data_validation'):
            validation = results['data_validation']
            if not validation['is_sufficient']:
                print("\n📊 Data validation issues:")
                for issue in validation['issues']:
                    print(f"   • {issue}")

if __name__ == "__main__":
    main()

"""
Base Trainer Class
Abstract base class for all ML training strategies with common functionality
"""

import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
import joblib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for all ML training strategies"""
    
    def __init__(self,
                 storage_manager=None,
                 min_rows_total: int = 500,
                 min_rows_per_symbol: int = 50,
                 min_time_span_hours: int = 6,
                 model_output_dir: str = "models",
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize base trainer
        
        Args:
            storage_manager: Storage manager for data access
            min_rows_total: Minimum total rows required for training
            min_rows_per_symbol: Minimum rows per symbol
            min_time_span_hours: Minimum time span of data in hours
            model_output_dir: Directory to save trained models
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility
        """
        self.storage_manager = storage_manager
        self.min_rows_total = min_rows_total
        self.min_rows_per_symbol = min_rows_per_symbol
        self.min_time_span_hours = min_time_span_hours
        self.model_output_dir = model_output_dir
        self.test_size = test_size
        self.random_state = random_state
        
        # Model state
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.target_column = 'future_return'
        
        # Training results
        self.training_results = {}
        self.is_trained = False
        
        # Ensure model directory exists
        os.makedirs(self.model_output_dir, exist_ok=True)
        
        logger.info(f"🤖 {self.__class__.__name__} initialized")
        logger.info(f"   Min rows total: {self.min_rows_total}")
        logger.info(f"   Min rows per symbol: {self.min_rows_per_symbol}")
        logger.info(f"   Min time span: {self.min_time_span_hours} hours")
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create and return the ML model instance"""
        pass
    
    @abstractmethod
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train the model on provided data
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary with training results
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that data meets minimum requirements for training
        
        Args:
            data: Combined market data from all sources
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_sufficient': True,
            'issues': [],
            'statistics': {},
            'data_quality': {}
        }
        
        try:
            if data.empty:
                validation_result['is_sufficient'] = False
                validation_result['issues'].append("No data provided")
                return validation_result
            
            # Basic statistics
            total_rows = len(data)
            symbols = data['symbol'].unique() if 'symbol' in data.columns else ['UNKNOWN']
            
            validation_result['statistics'] = {
                'total_rows': total_rows,
                'symbols': list(symbols),
                'symbol_count': len(symbols),
                'date_range': self._get_date_range(data),
                'time_span_hours': self._calculate_time_span_hours(data)
            }
            
            # Check minimum total rows
            if total_rows < self.min_rows_total:
                validation_result['is_sufficient'] = False
                validation_result['issues'].append(
                    f"Insufficient total rows: {total_rows} < {self.min_rows_total}"
                )
            
            # Check rows per symbol
            if 'symbol' in data.columns:
                symbol_counts = data['symbol'].value_counts()
                insufficient_symbols = symbol_counts[symbol_counts < self.min_rows_per_symbol]
                
                if not insufficient_symbols.empty:
                    validation_result['issues'].append(
                        f"Symbols with insufficient data: {dict(insufficient_symbols)}"
                    )
                
                validation_result['statistics']['rows_per_symbol'] = dict(symbol_counts)
            
            # Check time span
            time_span_hours = validation_result['statistics']['time_span_hours']
            if time_span_hours < self.min_time_span_hours:
                validation_result['is_sufficient'] = False
                validation_result['issues'].append(
                    f"Insufficient time span: {time_span_hours:.1f}h < {self.min_time_span_hours}h"
                )
            
            # Check data quality
            validation_result['data_quality'] = self._assess_data_quality(data)
            
            # Overall assessment
            if validation_result['issues']:
                validation_result['is_sufficient'] = len(validation_result['issues']) == 0
            
            logger.info(f"📊 Data validation: {total_rows} rows, {len(symbols)} symbols")
            if validation_result['is_sufficient']:
                logger.info("✅ Data validation passed")
            else:
                logger.warning(f"⚠️ Data validation issues: {len(validation_result['issues'])}")
                for issue in validation_result['issues']:
                    logger.warning(f"   • {issue}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Error in data validation: {e}")
            validation_result['is_sufficient'] = False
            validation_result['issues'].append(f"Validation error: {e}")
            return validation_result
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training by creating features and targets
        
        Args:
            data: Raw market data
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        try:
            logger.info("🔧 Preparing training data...")
            
            # Sort by timestamp to ensure proper ordering
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Create target variable (future return)
            data = self._create_target_variable(data)
            
            # Engineer features
            data = self._engineer_features(data)
            
            # Select feature columns
            feature_cols = self._select_feature_columns(data)
            self.feature_columns = feature_cols
            
            # Prepare feature matrix and target vector
            X = data[feature_cols].copy()
            y = data[self.target_column].copy()
            
            # Remove rows with missing targets
            valid_mask = ~y.isna()
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask].reset_index(drop=True)
            
            # Handle missing features
            X = X.fillna(method='forward').fillna(method='backward').fillna(0)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            logger.info(f"✅ Prepared {len(X)} samples with {len(feature_cols)} features")
            logger.info(f"   Target variable: {self.target_column}")
            logger.info(f"   Feature columns: {len(feature_cols)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """
        Run complete training pipeline from data collection to model saving
        
        Returns:
            Dictionary with pipeline results
        """
        try:
            logger.info("🚀 Starting full training pipeline...")
            
            pipeline_result = {
                'pipeline_success': False,
                'data_validation': {},
                'training_results': {},
                'model_path': None,
                'performance_metrics': {}
            }
            
            # Step 1: Collect and validate data
            logger.info("📊 Step 1: Data collection and validation")
            data = self._collect_training_data()
            
            if data.empty:
                pipeline_result['data_validation'] = {
                    'is_sufficient': False,
                    'issues': ['No data collected']
                }
                return pipeline_result
            
            validation_result = self.validate_data(data)
            pipeline_result['data_validation'] = validation_result
            
            if not validation_result['is_sufficient']:
                logger.error("❌ Data validation failed")
                return pipeline_result
            
            # Step 2: Prepare training data
            logger.info("🔧 Step 2: Data preparation")
            X, y = self.prepare_training_data(data)
            
            if X.empty or y.empty:
                pipeline_result['data_validation']['issues'].append('Data preparation failed')
                return pipeline_result
            
            # Step 3: Split data
            logger.info("📊 Step 3: Train/test split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=False  # Preserve temporal order
            )
            
            # Step 4: Scale features
            logger.info("⚖️ Step 4: Feature scaling")
            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Step 5: Train model
            logger.info("🤖 Step 5: Model training")
            self.model = self.create_model()
            training_results = self.train_model(X_train_scaled, y_train)
            
            # Step 6: Evaluate model
            logger.info("📈 Step 6: Model evaluation")
            y_train_pred = self.predict(X_train_scaled)
            y_test_pred = self.predict(X_test_scaled)
            
            performance_metrics = self._evaluate_model(
                y_train, y_train_pred, y_test, y_test_pred
            )
            
            # Step 7: Save model
            logger.info("💾 Step 7: Model saving")
            model_path = self._save_model()
            
            # Compile results
            pipeline_result.update({
                'pipeline_success': True,
                'training_results': {
                    **training_results,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features_used': len(self.feature_columns),
                    'model_path': model_path,
                    'performance': performance_metrics
                },
                'model_path': model_path,
                'performance_metrics': performance_metrics
            })
            
            self.is_trained = True
            logger.info("✅ Training pipeline completed successfully!")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            pipeline_result['training_results'] = {'error': str(e)}
            return pipeline_result
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from file
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Load model components
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.training_results = model_data.get('training_results', {})
            
            self.is_trained = True
            
            logger.info(f"✅ Model loaded successfully from {model_path}")
            logger.info(f"   Features: {len(self.feature_columns)}")
            logger.info(f"   Model type: {type(self.model).__name__}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def _collect_training_data(self) -> pd.DataFrame:
        """Collect data from storage for training"""
        try:
            if not self.storage_manager:
                logger.error("❌ No storage manager provided")
                return pd.DataFrame()
            
            # Get all available data
            all_data = []
            
            # Try to get data from memory first (fastest)
            memory_data = self.storage_manager.get_all_memory_data()
            if memory_data:
                logger.info(f"📊 Found {len(memory_data)} rows in memory")
                all_data.extend(memory_data)
            
            # Try to get data from local storage
            local_files = self.storage_manager.list_local_files()
            if local_files:
                logger.info(f"📁 Found {len(local_files)} local files")
                for file_path in local_files[:10]:  # Limit to recent files
                    try:
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            all_data.append(df)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not read {file_path}: {e}")
            
            # Combine all data
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # Remove duplicates
                if 'timestamp' in combined_data.columns and 'symbol' in combined_data.columns:
                    combined_data = combined_data.drop_duplicates(
                        subset=['timestamp', 'symbol'], keep='last'
                    )
                
                logger.info(f"📊 Collected {len(combined_data)} total rows")
                return combined_data
            else:
                logger.warning("⚠️ No data found in storage")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error collecting training data: {e}")
            return pd.DataFrame()
    
    def _create_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for prediction"""
        try:
            data = data.copy()
            
            # Create future return target (predict next period return)
            if 'close' in data.columns:
                # Simple return prediction
                data['future_return'] = data.groupby('symbol')['close'].pct_change().shift(-1)
                
                # Alternative: predict price direction (classification target)
                # data['future_direction'] = (data['future_return'] > 0).astype(int)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error creating target variable: {e}")
            return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML training"""
        try:
            # Import feature engineering utilities
            from .feature_engineer import FeatureEngineer
            
            engineer = FeatureEngineer()
            return engineer.engineer_features(data)
            
        except ImportError:
            # Fallback to basic feature engineering
            return self._basic_feature_engineering(data)
        except Exception as e:
            logger.error(f"❌ Error in feature engineering: {e}")
            return data
    
    def _basic_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic feature engineering fallback"""
        try:
            data = data.copy()
            
            # Price-based features
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                data['price_range'] = (data['high'] - data['low']) / data['close']
                data['body_size'] = abs(data['close'] - data['open']) / data['close']
                data['upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['close']
                data['lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['close']
            
            # Volume features
            if 'volume' in data.columns:
                data['volume_ma'] = data.groupby('symbol')['volume'].rolling(20).mean().reset_index(0, drop=True)
                data['volume_ratio'] = data['volume'] / data['volume_ma']
            
            # Simple moving averages
            if 'close' in data.columns:
                for window in [5, 10, 20]:
                    data[f'sma_{window}'] = data.groupby('symbol')['close'].rolling(window).mean().reset_index(0, drop=True)
                    data[f'price_vs_sma_{window}'] = data['close'] / data[f'sma_{window}'] - 1
            
            # Price changes
            if 'close' in data.columns:
                data['price_change_1'] = data.groupby('symbol')['close'].pct_change(1)
                data['price_change_5'] = data.groupby('symbol')['close'].pct_change(5)
                data['volatility'] = data.groupby('symbol')['price_change_1'].rolling(20).std().reset_index(0, drop=True)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error in basic feature engineering: {e}")
            return data
    
    def _select_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Select appropriate feature columns for training"""
        try:
            # Exclude non-feature columns
            exclude_cols = {
                'timestamp', 'symbol', 'close_time', 
                'future_return', 'future_direction',  # target variables
                'quote_asset_volume', 'number_of_trades',  # often not useful
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            }
            
            # Select numeric columns that aren't excluded
            feature_cols = []
            for col in data.columns:
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col]):
                    feature_cols.append(col)
            
            logger.info(f"📊 Selected {len(feature_cols)} feature columns")
            return feature_cols
            
        except Exception as e:
            logger.error(f"❌ Error selecting feature columns: {e}")
            return []
    
    def _evaluate_model(self, y_train, y_train_pred, y_test, y_test_pred) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            metrics = {}
            
            # Training metrics
            metrics['train_mse'] = mean_squared_error(y_train, y_train_pred)
            metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
            metrics['train_r2'] = r2_score(y_train, y_train_pred)
            
            # Test metrics
            metrics['test_mse'] = mean_squared_error(y_test, y_test_pred)
            metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
            metrics['test_r2'] = r2_score(y_test, y_test_pred)
            
            # Overfitting check
            metrics['overfitting_score'] = metrics['train_r2'] - metrics['test_r2']
            
            logger.info("📈 Model Performance:")
            logger.info(f"   Train R²: {metrics['train_r2']:.4f}")
            logger.info(f"   Test R²: {metrics['test_r2']:.4f}")
            logger.info(f"   Test MSE: {metrics['test_mse']:.6f}")
            logger.info(f"   Overfitting: {metrics['overfitting_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error evaluating model: {e}")
            return {}
    
    def _save_model(self) -> str:
        """Save trained model to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"production_model_{self.__class__.__name__.lower()}_{timestamp}.joblib"
            model_path = os.path.join(self.model_output_dir, model_filename)
            
            # Prepare model data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'training_results': self.training_results,
                'model_type': self.__class__.__name__,
                'created_at': datetime.now().isoformat(),
                'target_column': self.target_column
            }
            
            # Save model
            joblib.dump(model_data, model_path)
            
            logger.info(f"💾 Model saved: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return ""
    
    def _get_date_range(self, data: pd.DataFrame) -> Dict[str, str]:
        """Get date range of data"""
        try:
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                return {
                    'start': data['timestamp'].min().isoformat(),
                    'end': data['timestamp'].max().isoformat()
                }
            else:
                return {'start': 'Unknown', 'end': 'Unknown'}
        except:
            return {'start': 'Unknown', 'end': 'Unknown'}
    
    def _calculate_time_span_hours(self, data: pd.DataFrame) -> float:
        """Calculate time span of data in hours"""
        try:
            if 'timestamp' in data.columns and len(data) > 1:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                time_span = data['timestamp'].max() - data['timestamp'].min()
                return time_span.total_seconds() / 3600
            else:
                return 0.0
        except:
            return 0.0
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics"""
        try:
            quality_metrics = {
                'missing_values': data.isnull().sum().to_dict(),
                'duplicate_rows': data.duplicated().sum(),
                'infinite_values': {},
                'data_types': data.dtypes.astype(str).to_dict()
            }
            
            # Check for infinite values in numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                inf_count = np.isinf(data[col]).sum()
                if inf_count > 0:
                    quality_metrics['infinite_values'][col] = inf_count
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"❌ Error assessing data quality: {e}")
            return {}

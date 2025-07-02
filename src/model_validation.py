#!/usr/bin/env python3
"""
Production Model Validator - Strict validation for trading models
Ensures only real, production-ready models are used for trading
"""
import os
import logging
import joblib
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

class ProductionModelValidator:
    """Validates models are production-ready before trading"""
    
    def __init__(self, models_dir: str = None):
        from src.config import MODELS_DIR
        self.models_dir = Path(models_dir or MODELS_DIR)
        
        # Track validated models
        self.validated_models = {}
        self.model_cache = {}
        
        logger.info(f"🔍 Production Model Validator initialized")
        logger.info(f"   📁 Models directory: {self.models_dir}")
    
    def find_latest_production_model(self) -> Optional[Dict[str, Path]]:
        """Find the latest production ensemble model files"""
        try:
            # Look for production ensemble files
            ensemble_files = list(self.models_dir.glob("production_ensemble_*.joblib"))
            scaler_files = list(self.models_dir.glob("production_scaler_*.joblib"))
            feature_files = list(self.models_dir.glob("production_features_*.json"))
            
            if not (ensemble_files and scaler_files and feature_files):
                logger.warning(f"⚠️ No complete production model set found in {self.models_dir}")
                return None
            
            # Get the most recent files (by modification time)
            latest_ensemble = max(ensemble_files, key=lambda x: x.stat().st_mtime)
            latest_scaler = max(scaler_files, key=lambda x: x.stat().st_mtime)
            latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
            
            model_files = {
                'ensemble': latest_ensemble,
                'scaler': latest_scaler,
                'features': latest_features
            }
            
            logger.info(f"📊 Found latest production model:")
            logger.info(f"   🤖 Ensemble: {latest_ensemble.name}")
            logger.info(f"   🔧 Scaler: {latest_scaler.name}")
            logger.info(f"   📋 Features: {latest_features.name}")
            
            return model_files
            
        except Exception as e:
            logger.error(f"❌ Error finding production model: {e}")
            return None
    
    def validate_model_integrity(self, model_files: Dict[str, Path]) -> Dict[str, Any]:
        """Comprehensive model integrity validation"""
        validation = {
            'timestamp': datetime.now().isoformat(),
            'is_valid': False,
            'model_info': {},
            'validation_errors': [],
            'warnings': [],
            'tests_passed': 0,
            'tests_failed': 0
        }
        
        logger.info("🔍 VALIDATING MODEL INTEGRITY")
        
        try:
            # Test 1: Load ensemble model
            ensemble_path = model_files['ensemble']
            try:
                ensemble_model = joblib.load(ensemble_path)
                validation['model_info']['ensemble_type'] = type(ensemble_model).__name__
                validation['model_info']['ensemble_size'] = len(ensemble_model.estimators_) if hasattr(ensemble_model, 'estimators_') else 1
                validation['tests_passed'] += 1
                logger.info(f"✅ Ensemble model loaded: {type(ensemble_model).__name__}")
            except Exception as e:
                validation['validation_errors'].append(f"Failed to load ensemble: {e}")
                validation['tests_failed'] += 1
                logger.error(f"❌ Ensemble load failed: {e}")
                return validation
            
            # Test 2: Load scaler
            scaler_path = model_files['scaler']
            try:
                scaler = joblib.load(scaler_path)
                validation['model_info']['scaler_type'] = type(scaler).__name__
                validation['model_info']['scaler_features'] = len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else getattr(scaler, 'n_features_in_', 'unknown')
                validation['tests_passed'] += 1
                logger.info(f"✅ Scaler loaded: {type(scaler).__name__}")
            except Exception as e:
                validation['validation_errors'].append(f"Failed to load scaler: {e}")
                validation['tests_failed'] += 1
                logger.error(f"❌ Scaler load failed: {e}")
                return validation
            
            # Test 3: Load features
            features_path = model_files['features']
            try:
                with open(features_path, 'r') as f:
                    features = json.load(f)
                
                if not isinstance(features, list) or len(features) == 0:
                    raise ValueError("Features file is empty or invalid format")
                
                validation['model_info']['feature_count'] = len(features)
                validation['model_info']['sample_features'] = features[:5]
                validation['tests_passed'] += 1
                logger.info(f"✅ Features loaded: {len(features)} features")
            except Exception as e:
                validation['validation_errors'].append(f"Failed to load features: {e}")
                validation['tests_failed'] += 1
                logger.error(f"❌ Features load failed: {e}")
                return validation
            
            # Test 4: Feature consistency
            try:
                expected_features = len(features)
                scaler_features = getattr(scaler, 'n_features_in_', None)
                
                if scaler_features and scaler_features != expected_features:
                    validation['warnings'].append(f"Feature count mismatch: scaler expects {scaler_features}, features file has {expected_features}")
                    logger.warning(f"⚠️ Feature count mismatch: {scaler_features} vs {expected_features}")
                else:
                    validation['tests_passed'] += 1
                    logger.info("✅ Feature consistency verified")
            except Exception as e:
                validation['validation_errors'].append(f"Feature consistency check failed: {e}")
                validation['tests_failed'] += 1
            
            # Test 5: Model prediction test with dummy data
            try:
                # Create dummy data that matches expected features
                dummy_data = np.random.randn(1, len(features))
                
                # Scale dummy data
                dummy_scaled = scaler.transform(dummy_data)
                
                # Make prediction
                prediction = ensemble_model.predict(dummy_scaled)
                prob_prediction = ensemble_model.predict_proba(dummy_scaled)
                
                validation['model_info']['output_classes'] = len(prob_prediction[0])
                validation['model_info']['sample_prediction'] = int(prediction[0])
                validation['model_info']['sample_probabilities'] = [float(p) for p in prob_prediction[0]]
                
                validation['tests_passed'] += 1
                logger.info(f"✅ Prediction test passed: {validation['model_info']['output_classes']} classes")
                
            except Exception as e:
                validation['validation_errors'].append(f"Prediction test failed: {e}")
                validation['tests_failed'] += 1
                logger.error(f"❌ Prediction test failed: {e}")
                return validation
            
            # Test 6: File timestamps (ensure models are recent)
            try:
                model_age_hours = (datetime.now().timestamp() - ensemble_path.stat().st_mtime) / 3600
                
                if model_age_hours > 168:  # 7 days
                    validation['warnings'].append(f"Model is old: {model_age_hours:.1f} hours")
                    logger.warning(f"⚠️ Model age: {model_age_hours:.1f} hours")
                
                validation['model_info']['model_age_hours'] = model_age_hours
                validation['tests_passed'] += 1
                
            except Exception as e:
                validation['warnings'].append(f"Could not check model age: {e}")
            
            # Overall validation result
            if validation['tests_failed'] == 0:
                validation['is_valid'] = True
                logger.info("🎉 MODEL VALIDATION PASSED - Ready for production trading")
            else:
                logger.error(f"❌ MODEL VALIDATION FAILED - {validation['tests_failed']} tests failed")
            
            return validation
            
        except Exception as e:
            validation['validation_errors'].append(f"Validation process failed: {e}")
            logger.error(f"❌ Model validation process failed: {e}")
            return validation
    
    def load_validated_model(self) -> Optional[Dict[str, Any]]:
        """Load and cache a validated production model"""
        try:
            # Find latest model
            model_files = self.find_latest_production_model()
            if not model_files:
                logger.error("❌ No production model found")
                return None
            
            # Validate model
            validation = self.validate_model_integrity(model_files)
            if not validation['is_valid']:
                logger.error("❌ Model validation failed - cannot load for trading")
                for error in validation['validation_errors']:
                    logger.error(f"   • {error}")
                return None
            
            # Load validated model components
            ensemble_model = joblib.load(model_files['ensemble'])
            scaler = joblib.load(model_files['scaler'])
            
            with open(model_files['features'], 'r') as f:
                features = json.load(f)
            
            model_package = {
                'ensemble': ensemble_model,
                'scaler': scaler,
                'features': features,
                'model_files': model_files,
                'validation': validation,
                'loaded_at': datetime.now().isoformat()
            }
            
            # Cache the model
            self.model_cache['production'] = model_package
            
            logger.info("✅ PRODUCTION MODEL LOADED AND CACHED")
            logger.info(f"   🤖 Model: {validation['model_info']['ensemble_type']}")
            logger.info(f"   ⚙️ Features: {len(features)}")
            logger.info(f"   🎯 Classes: {validation['model_info']['output_classes']}")
            
            return model_package
            
        except Exception as e:
            logger.error(f"❌ Failed to load validated model: {e}")
            return None
    
    def predict_with_validation(self, features_dict: Dict[str, float], confidence_threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """Make prediction with confidence validation"""
        try:
            # Ensure model is loaded
            if 'production' not in self.model_cache:
                model_package = self.load_validated_model()
                if not model_package:
                    return None
            else:
                model_package = self.model_cache['production']
            
            ensemble = model_package['ensemble']
            scaler = model_package['scaler']
            expected_features = model_package['features']
            
            # Prepare feature vector
            feature_vector = []
            missing_features = []
            
            for feature_name in expected_features:
                if feature_name in features_dict:
                    feature_vector.append(features_dict[feature_name])
                else:
                    feature_vector.append(0.0)  # Default value for missing features
                    missing_features.append(feature_name)
            
            if missing_features:
                logger.warning(f"⚠️ Missing features filled with 0.0: {len(missing_features)}")
            
            # Scale features
            feature_array = np.array(feature_vector).reshape(1, -1)
            scaled_features = scaler.transform(feature_array)
            
            # Make prediction
            prediction = ensemble.predict(scaled_features)[0]
            probabilities = ensemble.predict_proba(scaled_features)[0]
            
            # Calculate confidence
            max_confidence = max(probabilities)
            
            prediction_result = {
                'prediction': int(prediction),
                'probabilities': [float(p) for p in probabilities],
                'confidence': float(max_confidence),
                'is_confident': max_confidence >= confidence_threshold,
                'features_used': len(expected_features),
                'missing_features': len(missing_features),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log prediction details
            logger.info(f"📊 MODEL PREDICTION:")
            logger.info(f"   🎯 Class: {prediction}")
            logger.info(f"   🔢 Confidence: {max_confidence:.3f}")
            logger.info(f"   ✅ Meets threshold: {max_confidence >= confidence_threshold}")
            logger.info(f"   ⚙️ Features: {len(expected_features)} used, {len(missing_features)} missing")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'has_validated_model': 'production' in self.model_cache,
            'models_directory': str(self.models_dir),
            'available_models': []
        }
        
        # Check for available model files
        try:
            ensemble_files = list(self.models_dir.glob("production_ensemble_*.joblib"))
            status['available_models'] = [f.name for f in ensemble_files]
        except Exception as e:
            status['error'] = f"Could not check models directory: {e}"
        
        if 'production' in self.model_cache:
            model_info = self.model_cache['production']['validation']['model_info']
            status['loaded_model'] = {
                'type': model_info.get('ensemble_type', 'unknown'),
                'features': model_info.get('feature_count', 0),
                'classes': model_info.get('output_classes', 0),
                'loaded_at': self.model_cache['production']['loaded_at']
            }
        
        return status
        self._load_performance_history()
    
    def _load_model(self) -> bool:
        """Load the current model with validation"""
        try:
            model_paths = [
                f"data/models/random_forest/trained_model.pkl",
                f"data/models/xgboost/trained_model.pkl"
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.current_model = pickle.load(f)
                    
                    # Get model version from timestamp
                    mod_time = os.path.getmtime(model_path)
                    self.model_version = datetime.fromtimestamp(mod_time).strftime("%Y%m%d_%H%M%S")
                    self.model_load_time = datetime.utcnow()
                    
                    logger.info(f"✅ Model loaded: {model_path} (version: {self.model_version})")
                    model_loaded = True
                    break
            
            if not model_loaded:
                logger.error("❌ No valid model found")
                self.is_model_valid = False
                return False
            
            # Validate model has required methods
            if not hasattr(self.current_model, 'predict') or not hasattr(self.current_model, 'predict_proba'):
                logger.error("❌ Model missing required methods")
                self.is_model_valid = False
                return False
            
            self.is_model_valid = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_model_valid = False
            return False
    
    def _load_performance_history(self):
        """Load historical performance metrics"""
        try:
            history_file = f"data/models/performance_history_{self.model_name}.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                for entry in data:
                    metrics = ModelPerformanceMetrics(
                        timestamp=datetime.fromisoformat(entry['timestamp']),
                        accuracy=entry['accuracy'],
                        precision=entry['precision'],
                        recall=entry['recall'],
                        f1_score=entry['f1_score'],
                        win_rate=entry['win_rate'],
                        avg_profit=entry['avg_profit'],
                        total_trades=entry['total_trades'],
                        confidence_correlation=entry.get('confidence_correlation', 0.0),
                        model_version=entry.get('model_version', 'unknown')
                    )
                    self.performance_history.append(metrics)
                
                logger.info(f"📊 Loaded {len(self.performance_history)} performance records")
                
        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")
    
    def _save_performance_history(self):
        """Save performance history to disk"""
        try:
            os.makedirs("data/models", exist_ok=True)
            history_file = f"data/models/performance_history_{self.model_name}.json"
            
            data = []
            for metrics in self.performance_history:
                data.append({
                    'timestamp': metrics.timestamp.isoformat(),
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'win_rate': metrics.win_rate,
                    'avg_profit': metrics.avg_profit,
                    'total_trades': metrics.total_trades,
                    'confidence_correlation': metrics.confidence_correlation,
                    'model_version': metrics.model_version
                })
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")
    
    def calculate_current_performance(self) -> Optional[ModelPerformanceMetrics]:
        """Calculate current model performance from recent trades"""
        try:
            # Load recent trade data
            trades_file = f"data/transactions/{self.model_name}_trades.csv"
            if not os.path.exists(trades_file):
                logger.warning("No trade data found for performance calculation")
                return None
            
            df = pd.read_csv(trades_file)
            if len(df) < 10:  # Need at least 10 trades
                logger.warning(f"Insufficient trades for performance calculation: {len(df)}")
                return None
            
            # Get recent trades (last 50 or all if less)
            recent_df = df.tail(self.drift_detection_window)
            
            # Calculate basic metrics
            win_rate = recent_df['was_successful'].mean() * 100
            avg_profit = recent_df['pnl_percent'].mean()
            total_trades = len(recent_df)
            
            # Calculate ML metrics (if we have confidence data)
            if 'confidence' in recent_df.columns:
                # Use confidence as prediction probability and success as actual
                y_true = recent_df['was_successful'].astype(int)
                y_pred_proba = recent_df['confidence']
                y_pred = (y_pred_proba > 0.5).astype(int)  # Binary predictions
                
                if len(np.unique(y_true)) > 1:  # Need both classes for metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    
                    # Calculate confidence correlation with actual performance
                    confidence_correlation = np.corrcoef(recent_df['confidence'], recent_df['pnl_percent'])[0, 1]
                    if np.isnan(confidence_correlation):
                        confidence_correlation = 0.0
                else:
                    accuracy = precision = recall = f1 = 0.0
                    confidence_correlation = 0.0
            else:
                accuracy = precision = recall = f1 = 0.0
                confidence_correlation = 0.0
            
            metrics = ModelPerformanceMetrics(
                timestamp=datetime.utcnow(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                win_rate=win_rate,
                avg_profit=avg_profit,
                total_trades=total_trades,
                confidence_correlation=confidence_correlation,
                model_version=self.model_version or "unknown"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return None
    
    def detect_drift(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Detect if model performance has drifted"""
        try:
            current_metrics = self.calculate_current_performance()
            if not current_metrics:
                return True, "Cannot calculate performance metrics", {}
            
            # Add to history
            self.performance_history.append(current_metrics)
            self._save_performance_history()
            
            # Check absolute performance thresholds
            if current_metrics.win_rate < self.validation_threshold:
                return True, f"Win rate below threshold: {current_metrics.win_rate:.1f}% < {self.validation_threshold}%", {
                    "current_win_rate": current_metrics.win_rate,
                    "threshold": self.validation_threshold,
                    "total_trades": current_metrics.total_trades
                }
            
            # Check for trending performance decline
            if len(self.performance_history) >= 3:
                recent_3 = self.performance_history[-3:]
                win_rates = [m.win_rate for m in recent_3]
                
                # Check if consistently declining
                if len(win_rates) >= 3 and win_rates[-1] < win_rates[-2] < win_rates[-3]:
                    decline = win_rates[-3] - win_rates[-1]
                    if decline > 10:  # 10% decline
                        return True, f"Consistent performance decline: {decline:.1f}% drop", {
                            "win_rate_trend": win_rates,
                            "decline_percent": decline
                        }
            
            # Check confidence correlation (should be positive)
            if current_metrics.confidence_correlation < -0.2:
                return True, f"Poor confidence correlation: {current_metrics.confidence_correlation:.3f}", {
                    "confidence_correlation": current_metrics.confidence_correlation
                }
            
            # Check model age (retraining needed if > 7 days old)
            if self.model_load_time:
                model_age_days = (datetime.utcnow() - self.model_load_time).days
                if model_age_days > 7:
                    return True, f"Model too old: {model_age_days} days", {
                        "model_age_days": model_age_days
                    }
            
            return False, "Model performance is acceptable", {
                "current_win_rate": current_metrics.win_rate,
                "avg_profit": current_metrics.avg_profit,
                "confidence_correlation": current_metrics.confidence_correlation
            }
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return True, f"Drift detection error: {e}", {}
    
    def validate_model_for_trading(self) -> Tuple[bool, str]:
        """Validate if model is safe to use for trading"""
        if not self.is_model_valid:
            return False, "Model not loaded or invalid"
        
        # Check if model file still exists
        model_paths = [
            f"data/models/random_forest/trained_model.pkl",
            f"data/models/xgboost/trained_model.pkl"
        ]
        
        model_exists = any(os.path.exists(path) for path in model_paths)
        if not model_exists:
            return False, "Model file no longer exists"
        
        # Check for drift
        has_drift, drift_reason, _ = self.detect_drift()
        if has_drift:
            return False, f"Model drift detected: {drift_reason}"
        
        return True, "Model is valid for trading"
    
    def force_model_reload(self) -> bool:
        """Force reload of model (after retraining)"""
        logger.info("🔄 Force reloading model...")
        success = self._load_model()
        if success:
            logger.info("✅ Model reloaded successfully")
        else:
            logger.error("❌ Model reload failed")
        return success
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_metrics = self.calculate_current_performance()
        
        summary = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_valid": self.is_model_valid,
            "model_age_hours": None,
            "current_performance": None,
            "historical_count": len(self.performance_history),
            "drift_status": None
        }
        
        if self.model_load_time:
            model_age = datetime.utcnow() - self.model_load_time
            summary["model_age_hours"] = model_age.total_seconds() / 3600
        
        if current_metrics:
            summary["current_performance"] = {
                "win_rate": current_metrics.win_rate,
                "avg_profit": current_metrics.avg_profit,
                "total_trades": current_metrics.total_trades,
                "confidence_correlation": current_metrics.confidence_correlation
            }
        
        # Get drift status
        has_drift, drift_reason, drift_details = self.detect_drift()
        summary["drift_status"] = {
            "has_drift": has_drift,
            "reason": drift_reason,
            "details": drift_details
        }
        
        return summary

class ModelValidationService:
    """Service to manage multiple model validators"""
    
    def __init__(self):
        self.validators: Dict[str, ModelDriftDetector] = {}
        self.trading_enabled = True
        self.last_validation_time = None
    
    def register_model(self, model_name: str) -> ModelDriftDetector:
        """Register a model for validation"""
        if model_name not in self.validators:
            self.validators[model_name] = ModelDriftDetector(model_name)
            logger.info(f"📝 Registered model for validation: {model_name}")
        
        return self.validators[model_name]
    
    def validate_all_models(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate all registered models"""
        self.last_validation_time = datetime.utcnow()
        results = {}
        all_valid = True
        
        for model_name, validator in self.validators.items():
            is_valid, reason = validator.validate_model_for_trading()
            results[model_name] = {
                "valid": is_valid,
                "reason": reason,
                "performance": validator.get_performance_summary()
            }
            
            if not is_valid:
                all_valid = False
                logger.warning(f"⚠️ Model {model_name} invalid: {reason}")
        
        if not all_valid:
            self.trading_enabled = False
            logger.error("🚫 Trading disabled due to invalid models")
        else:
            self.trading_enabled = True
            logger.info("✅ All models validated for trading")
        
        return all_valid, results
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on model validation"""
        if not self.trading_enabled:
            return False, "Trading disabled due to model validation failures"
        
        # Check if validation is recent (within last hour)
        if self.last_validation_time:
            age = datetime.utcnow() - self.last_validation_time
            if age > timedelta(hours=1):
                return False, "Model validation is stale - revalidation needed"
        else:
            return False, "Models not yet validated"
        
        return True, "Models validated for trading"
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get overall validation status"""
        return {
            "trading_enabled": self.trading_enabled,
            "last_validation": self.last_validation_time.isoformat() if self.last_validation_time else None,
            "registered_models": list(self.validators.keys()),
            "validation_details": {name: validator.get_performance_summary() 
                                 for name, validator in self.validators.items()}
        }
    
    def record_prediction_result(self, model_name: str, prediction: float, confidence: float, outcome: bool, actual_pnl: float = None):
        """Record a prediction result for model performance tracking"""
        if model_name not in self.validators:
            logger.warning(f"Recording result for unregistered model: {model_name}")
            self.register_model(model_name)
        
        validator = self.validators[model_name]
        
        # Log the prediction result
        logger.info(f"📊 Recorded prediction: {model_name} | Pred: {prediction:.3f} | Conf: {confidence:.3f} | Success: {outcome}")
        
        # This could be expanded to maintain a rolling window of recent predictions
        # For now, we rely on the CSV files for persistence
        
        return True
    
    def get_model_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific model"""
        if model_name not in self.validators:
            return None
        
        validator = self.validators[model_name]
        current_metrics = validator.calculate_current_performance()
        
        if current_metrics:
            return {
                'accuracy': current_metrics.accuracy,
                'precision': current_metrics.precision,
                'recall': current_metrics.recall,
                'f1_score': current_metrics.f1_score,
                'win_rate': current_metrics.win_rate,
                'avg_profit': current_metrics.avg_profit,
                'total_trades': current_metrics.total_trades,
                'confidence_correlation': current_metrics.confidence_correlation
            }
        
        return None

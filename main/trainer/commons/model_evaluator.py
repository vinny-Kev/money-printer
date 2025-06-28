"""
Model Evaluator
Comprehensive model evaluation and validation utilities
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and validation"""
    
    def __init__(self, task_type: str = 'regression'):
        """
        Initialize model evaluator
        
        Args:
            task_type: Type of ML task ('regression' or 'classification')
        """
        self.task_type = task_type
        self.evaluation_results = {}
        logger.info(f"📊 ModelEvaluator initialized for {task_type}")
    
    def evaluate_model(self, 
                      model,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model instance
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info(f"📊 Evaluating {model_name}...")
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Basic metrics
            basic_metrics = self._calculate_basic_metrics(
                y_train, y_train_pred, y_test, y_test_pred
            )
            
            # Cross-validation
            cv_scores = self._perform_cross_validation(model, X_train, y_train)
            
            # Feature importance
            feature_importance = self._get_feature_importance(model, X_train.columns)
            
            # Additional analysis
            prediction_analysis = self._analyze_predictions(
                y_train, y_train_pred, y_test, y_test_pred
            )
            
            # Compile results
            evaluation_result = {
                'model_name': model_name,
                'task_type': self.task_type,
                'basic_metrics': basic_metrics,
                'cross_validation': cv_scores,
                'feature_importance': feature_importance,
                'prediction_analysis': prediction_analysis,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Store for later reference
            self.evaluation_results[model_name] = evaluation_result
            
            logger.info(f"✅ {model_name} evaluation completed")
            self._log_key_metrics(basic_metrics)
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_name}: {e}")
            return {}
    
    def compare_models(self, 
                      evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Model comparison summary
        """
        try:
            logger.info(f"🏆 Comparing {len(evaluation_results)} models...")
            
            if not evaluation_results:
                return {}
            
            # Extract key metrics for comparison
            comparison_data = []
            for result in evaluation_results:
                if 'basic_metrics' in result:
                    metrics = result['basic_metrics'].copy()
                    metrics['model_name'] = result.get('model_name', 'Unknown')
                    comparison_data.append(metrics)
            
            if not comparison_data:
                return {}
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            # Determine best model based on primary metric
            primary_metric = 'test_r2' if self.task_type == 'regression' else 'test_accuracy'
            
            if primary_metric in comparison_df.columns:
                best_model_idx = comparison_df[primary_metric].idxmax()
                best_model = comparison_df.iloc[best_model_idx]
            else:
                best_model = comparison_df.iloc[0]
            
            # Generate rankings
            rankings = self._generate_model_rankings(comparison_df)
            
            comparison_result = {
                'comparison_summary': comparison_df.to_dict('records'),
                'best_model': {
                    'name': best_model['model_name'],
                    'metrics': best_model.to_dict()
                },
                'rankings': rankings,
                'primary_metric': primary_metric,
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🏆 Best model: {best_model['model_name']} ({primary_metric}: {best_model[primary_metric]:.4f})")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"❌ Error comparing models: {e}")
            return {}
    
    def validate_model_stability(self, 
                                model,
                                X: pd.DataFrame,
                                y: pd.Series,
                                n_splits: int = 5) -> Dict[str, Any]:
        """
        Validate model stability using time series cross-validation
        
        Args:
            model: Trained model instance
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV splits
            
        Returns:
            Stability validation results
        """
        try:
            logger.info(f"🔄 Validating model stability with {n_splits} splits...")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Track metrics across folds
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train on fold
                model.fit(X_fold_train, y_fold_train)
                
                # Predict on validation
                y_fold_pred = model.predict(X_fold_val)
                
                # Calculate metrics
                if self.task_type == 'regression':
                    fold_metric = {
                        'fold': fold + 1,
                        'mse': mean_squared_error(y_fold_val, y_fold_pred),
                        'mae': mean_absolute_error(y_fold_val, y_fold_pred),
                        'r2': r2_score(y_fold_val, y_fold_pred)
                    }
                else:
                    # Classification metrics would go here
                    fold_metric = {'fold': fold + 1}
                
                fold_metrics.append(fold_metric)
            
            # Analyze stability
            stability_analysis = self._analyze_stability(fold_metrics)
            
            result = {
                'fold_metrics': fold_metrics,
                'stability_analysis': stability_analysis,
                'n_splits': n_splits,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Stability validation completed")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in stability validation: {e}")
            return {}
    
    def generate_evaluation_report(self, 
                                  model_name: str = None,
                                  save_path: str = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            model_name: Specific model to report on (if None, reports all)
            save_path: Path to save report file
            
        Returns:
            Report text
        """
        try:
            if model_name and model_name in self.evaluation_results:
                results_to_report = {model_name: self.evaluation_results[model_name]}
            else:
                results_to_report = self.evaluation_results
            
            if not results_to_report:
                return "No evaluation results available."
            
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("MODEL EVALUATION REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Task Type: {self.task_type}")
            report_lines.append("")
            
            for name, result in results_to_report.items():
                report_lines.extend(self._format_model_report(name, result))
                report_lines.append("")
            
            # Summary if multiple models
            if len(results_to_report) > 1:
                report_lines.extend(self._format_summary_report(results_to_report))
            
            report_text = "\n".join(report_lines)
            
            # Save to file if requested
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"📄 Report saved to {save_path}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"❌ Error generating evaluation report: {e}")
            return f"Error generating report: {e}"
    
    def _calculate_basic_metrics(self, 
                               y_train, y_train_pred, 
                               y_test, y_test_pred) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        try:
            metrics = {}
            
            if self.task_type == 'regression':
                # Training metrics
                metrics['train_mse'] = mean_squared_error(y_train, y_train_pred)
                metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
                metrics['train_r2'] = r2_score(y_train, y_train_pred)
                metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
                
                # Test metrics
                metrics['test_mse'] = mean_squared_error(y_test, y_test_pred)
                metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
                metrics['test_r2'] = r2_score(y_test, y_test_pred)
                metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
                
                # Overfitting indicators
                metrics['overfitting_r2'] = metrics['train_r2'] - metrics['test_r2']
                metrics['overfitting_mse'] = metrics['test_mse'] / metrics['train_mse']
                
                # Additional regression metrics
                metrics['train_mape'] = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
                metrics['test_mape'] = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
                
            else:
                # Classification metrics would go here
                pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating basic metrics: {e}")
            return {}
    
    def _perform_cross_validation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation"""
        try:
            # Use TimeSeriesSplit for time series data
            cv = TimeSeriesSplit(n_splits=5)
            
            scoring = 'r2' if self.task_type == 'regression' else 'accuracy'
            
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            cv_results = {
                'scores': cv_scores.tolist(),
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'min_score': cv_scores.min(),
                'max_score': cv_scores.max(),
                'scoring_metric': scoring
            }
            
            return cv_results
            
        except Exception as e:
            logger.error(f"❌ Error in cross-validation: {e}")
            return {}
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """Extract feature importance from model"""
        try:
            importance_data = {}
            
            # Try to get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_data = {
                    'method': 'feature_importances_',
                    'importances': importance_df.to_dict('records'),
                    'top_features': importance_df.head(10)['feature'].tolist()
                }
                
            elif hasattr(model, 'coef_'):
                # Linear model coefficients
                coefficients = np.abs(model.coef_)
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coefficients
                }).sort_values('importance', ascending=False)
                
                importance_data = {
                    'method': 'coefficients',
                    'importances': importance_df.to_dict('records'),
                    'top_features': importance_df.head(10)['feature'].tolist()
                }
            
            return importance_data
            
        except Exception as e:
            logger.error(f"❌ Error extracting feature importance: {e}")
            return {}
    
    def _analyze_predictions(self, 
                           y_train, y_train_pred, 
                           y_test, y_test_pred) -> Dict[str, Any]:
        """Analyze prediction patterns"""
        try:
            analysis = {}
            
            if self.task_type == 'regression':
                # Residual analysis
                train_residuals = y_train - y_train_pred
                test_residuals = y_test - y_test_pred
                
                analysis['residual_analysis'] = {
                    'train_residual_mean': float(train_residuals.mean()),
                    'train_residual_std': float(train_residuals.std()),
                    'test_residual_mean': float(test_residuals.mean()),
                    'test_residual_std': float(test_residuals.std())
                }
                
                # Prediction distribution
                analysis['prediction_distribution'] = {
                    'train_pred_mean': float(y_train_pred.mean()),
                    'train_pred_std': float(y_train_pred.std()),
                    'test_pred_mean': float(y_test_pred.mean()),
                    'test_pred_std': float(y_test_pred.std()),
                    'train_actual_mean': float(y_train.mean()),
                    'test_actual_mean': float(y_test.mean())
                }
                
                # Prediction accuracy by magnitude
                analysis['accuracy_by_magnitude'] = self._analyze_accuracy_by_magnitude(
                    y_test, y_test_pred
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing predictions: {e}")
            return {}
    
    def _analyze_accuracy_by_magnitude(self, y_true, y_pred) -> Dict[str, float]:
        """Analyze prediction accuracy by target magnitude"""
        try:
            # Create magnitude bins
            y_true_abs = np.abs(y_true)
            bins = np.percentile(y_true_abs, [0, 25, 50, 75, 100])
            
            accuracy_by_bin = {}
            
            for i in range(len(bins) - 1):
                mask = (y_true_abs >= bins[i]) & (y_true_abs < bins[i + 1])
                if mask.sum() > 0:
                    bin_r2 = r2_score(y_true[mask], y_pred[mask])
                    bin_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                    
                    accuracy_by_bin[f'bin_{i+1}'] = {
                        'range': f'[{bins[i]:.4f}, {bins[i+1]:.4f})',
                        'count': int(mask.sum()),
                        'r2': float(bin_r2),
                        'mae': float(bin_mae)
                    }
            
            return accuracy_by_bin
            
        except Exception as e:
            logger.error(f"❌ Error analyzing accuracy by magnitude: {e}")
            return {}
    
    def _analyze_stability(self, fold_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze model stability across folds"""
        try:
            if not fold_metrics:
                return {}
            
            # Extract metric values
            metrics_df = pd.DataFrame(fold_metrics)
            
            stability_analysis = {}
            
            for metric in ['mse', 'mae', 'r2']:
                if metric in metrics_df.columns:
                    values = metrics_df[metric]
                    stability_analysis[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'cv': float(values.std() / values.mean()) if values.mean() != 0 else 0
                    }
            
            # Overall stability score (lower CV = more stable)
            if 'r2' in stability_analysis:
                stability_score = 1 / (1 + stability_analysis['r2']['cv'])
            else:
                stability_score = 0.5
            
            stability_analysis['overall_stability_score'] = stability_score
            
            return stability_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing stability: {e}")
            return {}
    
    def _generate_model_rankings(self, comparison_df: pd.DataFrame) -> Dict[str, List]:
        """Generate model rankings based on different metrics"""
        try:
            rankings = {}
            
            ranking_metrics = ['test_r2', 'test_mse', 'test_mae'] if self.task_type == 'regression' else []
            
            for metric in ranking_metrics:
                if metric in comparison_df.columns:
                    if 'r2' in metric:  # Higher is better
                        sorted_df = comparison_df.sort_values(metric, ascending=False)
                    else:  # Lower is better
                        sorted_df = comparison_df.sort_values(metric, ascending=True)
                    
                    rankings[metric] = [
                        {
                            'rank': i + 1,
                            'model': row['model_name'],
                            'value': row[metric]
                        }
                        for i, (_, row) in enumerate(sorted_df.iterrows())
                    ]
            
            return rankings
            
        except Exception as e:
            logger.error(f"❌ Error generating rankings: {e}")
            return {}
    
    def _format_model_report(self, model_name: str, result: Dict[str, Any]) -> List[str]:
        """Format individual model report section"""
        lines = []
        lines.append(f"Model: {model_name}")
        lines.append("-" * 40)
        
        # Basic metrics
        if 'basic_metrics' in result:
            metrics = result['basic_metrics']
            lines.append("Performance Metrics:")
            
            if self.task_type == 'regression':
                lines.append(f"  Train R²: {metrics.get('train_r2', 0):.4f}")
                lines.append(f"  Test R²:  {metrics.get('test_r2', 0):.4f}")
                lines.append(f"  Test MSE: {metrics.get('test_mse', 0):.6f}")
                lines.append(f"  Test MAE: {metrics.get('test_mae', 0):.6f}")
                lines.append(f"  Overfitting (R² diff): {metrics.get('overfitting_r2', 0):.4f}")
        
        # Cross-validation
        if 'cross_validation' in result:
            cv = result['cross_validation']
            lines.append(f"Cross-Validation ({cv.get('scoring_metric', 'unknown')}):")
            lines.append(f"  Mean: {cv.get('mean_score', 0):.4f} ± {cv.get('std_score', 0):.4f}")
            lines.append(f"  Range: [{cv.get('min_score', 0):.4f}, {cv.get('max_score', 0):.4f}]")
        
        # Top features
        if 'feature_importance' in result:
            importance = result['feature_importance']
            if 'top_features' in importance:
                lines.append("Top Features:")
                for i, feature in enumerate(importance['top_features'][:5]):
                    lines.append(f"  {i+1}. {feature}")
        
        return lines
    
    def _format_summary_report(self, results: Dict[str, Any]) -> List[str]:
        """Format summary report section"""
        lines = []
        lines.append("SUMMARY COMPARISON")
        lines.append("-" * 40)
        
        # Extract comparison data
        comparison_data = []
        for name, result in results.items():
            if 'basic_metrics' in result:
                data = {'model': name}
                data.update(result['basic_metrics'])
                comparison_data.append(data)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            # Best performing model
            if 'test_r2' in df.columns:
                best_idx = df['test_r2'].idxmax()
                best_model = df.iloc[best_idx]
                lines.append(f"Best Model (Test R²): {best_model['model']} ({best_model['test_r2']:.4f})")
            
            # Performance range
            if 'test_r2' in df.columns:
                lines.append(f"R² Range: [{df['test_r2'].min():.4f}, {df['test_r2'].max():.4f}]")
        
        return lines
    
    def _log_key_metrics(self, metrics: Dict[str, float]):
        """Log key metrics to console"""
        if self.task_type == 'regression':
            logger.info(f"   📈 Test R²: {metrics.get('test_r2', 0):.4f}")
            logger.info(f"   📉 Test MSE: {metrics.get('test_mse', 0):.6f}")
            logger.info(f"   ⚖️ Overfitting: {metrics.get('overfitting_r2', 0):.4f}")
        else:
            # Classification metrics would go here
            pass

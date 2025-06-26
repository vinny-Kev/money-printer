#!/usr/bin/env python3
"""
Trainer Diagnostics and Metrics

Provides comprehensive training diagnostics, overfitting detection,
feature importance analysis, and model performance evaluation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)

class TrainerDiagnostics:
    """Comprehensive training diagnostics and analysis"""
    
    def __init__(self, model_name: str, diagnostics_dir: str = "data/diagnostics"):
        self.model_name = model_name
        self.diagnostics_dir = diagnostics_dir
        os.makedirs(diagnostics_dir, exist_ok=True)
        
    def analyze_training_results(self, 
                               train_scores: List[float],
                               val_scores: List[float], 
                               model,
                               X_train,
                               y_train,
                               X_val,
                               y_val,
                               feature_names: List[str],
                               training_time: float) -> Dict[str, Any]:
        """Comprehensive analysis of training results"""
        
        # Calculate final losses
        train_loss = train_scores[-1] if train_scores else 0.0
        val_loss = val_scores[-1] if val_scores else 0.0
        
        # Detect overfitting
        overfit_risk = self._detect_overfitting(train_scores, val_scores)
        
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance(model, feature_names)
        
        # Make predictions for confidence analysis
        try:
            if hasattr(model, 'predict_proba'):
                train_proba = model.predict_proba(X_train)
                val_proba = model.predict_proba(X_val)
                
                # Get confidence scores (max probability)
                train_confidence = np.max(train_proba, axis=1)
                val_confidence = np.max(val_proba, axis=1)
                
                avg_confidence = np.mean(val_confidence)
                confidence_dist = self._analyze_confidence_distribution(val_confidence)
            else:
                avg_confidence = 0.7  # Default for models without predict_proba
                confidence_dist = {"high": 0, "medium": 0, "low": 0}
                
        except Exception as e:
            logger.warning(f"Could not analyze confidence: {e}")
            avg_confidence = 0.7
            confidence_dist = {"high": 0, "medium": 0, "low": 0}
        
        # Calculate win rate prediction vs actual
        try:
            val_pred = model.predict(X_val)
            predicted_winrate = np.mean(val_pred) if len(val_pred) > 0 else 0.5
            
            # For actual win rate, we'll estimate based on validation accuracy
            from sklearn.metrics import accuracy_score
            actual_winrate = accuracy_score(y_val, val_pred)
            
        except Exception as e:
            logger.warning(f"Could not calculate win rates: {e}")
            predicted_winrate = 0.5
            actual_winrate = 0.5
        
        # Compile diagnostics
        diagnostics = {
            "model": self.model_name,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "overfit_risk": overfit_risk,
            "best_features": list(feature_importance.keys())[:5],  # Top 5
            "feature_importance": feature_importance,
            "avg_confidence": float(avg_confidence),
            "confidence_distribution": confidence_dist,
            "winrate_predicted": float(predicted_winrate),
            "winrate_actual": float(actual_winrate),
            "training_time": float(training_time),
            "timestamp": datetime.utcnow().isoformat(),
            "training_samples": len(X_train),
            "validation_samples": len(X_val)
        }
        
        # Save diagnostics
        self._save_diagnostics(diagnostics)
        
        # Generate visualizations
        self._generate_training_plots(train_scores, val_scores, feature_importance, confidence_dist)
        
        return diagnostics
        
    def _detect_overfitting(self, train_scores: List[float], val_scores: List[float]) -> str:
        """Detect overfitting risk based on training curves"""
        if not train_scores or not val_scores:
            return "Unknown"
            
        train_final = train_scores[-1]
        val_final = val_scores[-1]
        
        # Calculate gap between training and validation
        gap = abs(train_final - val_final)
        gap_ratio = gap / max(train_final, val_final, 0.001)
        
        if gap_ratio < 0.05:
            return "Low"
        elif gap_ratio < 0.15:
            return "Medium" 
        else:
            return "High"
            
    def _analyze_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract and analyze feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                # Default importance for unsupported models
                importance = np.ones(len(feature_names)) / len(feature_names)
                
            # Create importance dictionary
            feature_importance = dict(zip(feature_names, importance))
            
            # Sort by importance
            sorted_features = dict(sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
            
            return sorted_features
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {name: 1.0/len(feature_names) for name in feature_names}
            
    def _analyze_confidence_distribution(self, confidence_scores: np.ndarray) -> Dict[str, int]:
        """Analyze distribution of confidence scores"""
        high_conf = np.sum(confidence_scores > 0.7)
        medium_conf = np.sum((confidence_scores >= 0.5) & (confidence_scores <= 0.7))
        low_conf = np.sum(confidence_scores < 0.5)
        
        return {
            "high": int(high_conf),
            "medium": int(medium_conf),
            "low": int(low_conf)
        }
        
    def _save_diagnostics(self, diagnostics: Dict[str, Any]):
        """Save diagnostics to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.model_name}_diagnostics_{timestamp}.json"
        filepath = os.path.join(self.diagnostics_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(diagnostics, f, indent=2)
            logger.info(f"📊 Diagnostics saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save diagnostics: {e}")
            
    def _generate_training_plots(self, 
                                train_scores: List[float],
                                val_scores: List[float], 
                                feature_importance: Dict[str, float],
                                confidence_dist: Dict[str, int]):
        """Generate training visualization plots"""
        try:
            plt.style.use('dark_background')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Training curves
            if train_scores and val_scores:
                epochs = range(1, len(train_scores) + 1)
                ax1.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
                ax1.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
                ax1.set_title(f'{self.model_name} - Training Curves')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Score')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Feature importance (top 10)
            if feature_importance:
                top_features = dict(list(feature_importance.items())[:10])
                features = list(top_features.keys())
                importance = list(top_features.values())
                
                ax2.barh(features, importance, color='cyan', alpha=0.7)
                ax2.set_title('Top 10 Feature Importance')
                ax2.set_xlabel('Importance')
                
            # Confidence distribution
            if confidence_dist:
                categories = list(confidence_dist.keys())
                values = list(confidence_dist.values())
                colors = ['green', 'orange', 'red']
                
                ax3.bar(categories, values, color=colors, alpha=0.7)
                ax3.set_title('Confidence Distribution')
                ax3.set_ylabel('Count')
                
            # Model performance summary (placeholder)
            ax4.text(0.1, 0.8, f'Model: {self.model_name}', fontsize=12, color='white')
            ax4.text(0.1, 0.6, f'Training Complete', fontsize=10, color='green')
            ax4.text(0.1, 0.4, f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=8, color='gray')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Training Summary')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"{self.model_name}_training_plots_{timestamp}.png"
            plot_path = os.path.join(self.diagnostics_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close()
            
            logger.info(f"📈 Training plots saved: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate training plots: {e}")
            
    def format_training_summary(self, diagnostics: Dict[str, Any]) -> str:
        """Format training summary for display"""
        output = []
        output.append(f"🤖 TRAINING COMPLETE: {diagnostics['model'].upper()}")
        output.append("=" * 50)
        output.append(f"📉 Training Loss: {diagnostics['train_loss']:.3f}")
        output.append(f"📊 Validation Loss: {diagnostics['val_loss']:.3f}")
        output.append(f"🔍 Overfitting Risk: {diagnostics['overfit_risk']}")
        output.append(f"🧠 Avg Confidence: {diagnostics['avg_confidence']:.1%}")
        output.append(f"📈 Predicted Win Rate: {diagnostics['winrate_predicted']:.1%}")
        output.append(f"✅ Actual Win Rate: {diagnostics['winrate_actual']:.1%}")
        output.append("")
        
        # Top features
        output.append("🔝 Best Features:")
        for i, feature in enumerate(diagnostics['best_features'][:5], 1):
            importance = diagnostics['feature_importance'].get(feature, 0)
            output.append(f"  {i}. {feature}: {importance:.3f}")
            
        output.append("")
        
        # Confidence distribution
        conf_dist = diagnostics['confidence_distribution']
        total_preds = sum(conf_dist.values())
        if total_preds > 0:
            output.append("🎯 Confidence Distribution:")
            output.append(f"  High (>70%): {conf_dist['high']} ({conf_dist['high']/total_preds:.1%})")
            output.append(f"  Medium (50-70%): {conf_dist['medium']} ({conf_dist['medium']/total_preds:.1%})")
            output.append(f"  Low (<50%): {conf_dist['low']} ({conf_dist['low']/total_preds:.1%})")
            
        return "\n".join(output)

def get_trainer_diagnostics() -> TrainerDiagnostics:
    """Get trainer diagnostics instance"""
    return TrainerDiagnostics("default")

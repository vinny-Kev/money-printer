#!/usr/bin/env python3
"""
Incremental Model Training System

This module provides functionality to retrain models using completed trade data.
It reads transaction logs and updates models based on actual trading outcomes.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.discord_notifications import send_trainer_notification

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncrementalTrainer:
    def __init__(self, model_name="random_forest_v1", min_trades=20):
        self.model_name = model_name
        self.min_trades = min_trades
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.transactions_dir = os.path.join(self.base_dir, "data", "transactions")
        self.models_dir = os.path.join(self.base_dir, "data", "models", "random_forest")
        self.training_csv_path = os.path.join(self.transactions_dir, f"{model_name}_trades.csv")
        
        # Ensure directories exist
        os.makedirs(self.transactions_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_trade_data(self, recent_n_trades=None):
        """
        Load trade data from CSV file.
        
        Args:
            recent_n_trades (int): If specified, only load the most recent N trades
            
        Returns:
            pd.DataFrame: Loaded trade data
        """
        if not os.path.exists(self.training_csv_path):
            logger.warning(f"No training data found at {self.training_csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.training_csv_path)
        
        if recent_n_trades:
            df = df.tail(recent_n_trades)
            logger.info(f"Loaded {len(df)} recent trades for analysis")
        else:
            logger.info(f"Loaded {len(df)} total trades for analysis")
        
        return df
    
    def calculate_win_rate(self, recent_n_trades=50):
        """
        Calculate win rate for recent trades.
        
        Args:
            recent_n_trades (int): Number of recent trades to analyze
            
        Returns:
            float: Win rate as a percentage (0-100)
        """
        df = self.load_trade_data(recent_n_trades)
        
        if len(df) < self.min_trades:
            logger.warning(f"Insufficient trades ({len(df)}) for win rate calculation")
            return 0.0
        
        win_rate = (df['was_successful'].sum() / len(df)) * 100
        logger.info(f"Win rate for recent {len(df)} trades: {win_rate:.2f}%")
        return win_rate
    
    def prepare_features_and_labels(self, df):
        """
        Convert trade data to features and labels for training.
        
        Args:
            df (pd.DataFrame): Trade data
            
        Returns:
            tuple: (X, y) features and labels
        """
        # Define features for training
        feature_columns = [
            'confidence', 'predicted_profit_pct', 'rsi_at_buy', 
            'macd_at_buy', 'volume_change_at_buy'
        ]
        
        # Filter to only include available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            logger.error("No valid features found in trade data")
            return None, None
        
        # Prepare features (X)
        X = df[available_features].fillna(0)
        
        # Prepare labels (y) - predict if trade will be successful
        y = df['was_successful'].astype(int)
        
        logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")
        logger.info(f"Features used: {available_features}")
        
        return X, y
    
    def should_retrain(self, win_rate_threshold=60.0, min_trades_required=None):
        """
        Determine if model should be retrained based on recent performance.
        
        Args:
            win_rate_threshold (float): Minimum win rate to skip retraining
            min_trades_required (int): Minimum trades needed before retraining
            
        Returns:
            tuple: (should_retrain: bool, reason: str)
        """
        if min_trades_required is None:
            min_trades_required = self.min_trades
        
        df = self.load_trade_data()
        
        if len(df) < min_trades_required:
            return False, f"Insufficient trades ({len(df)}) for retraining (need {min_trades_required})"
        
        recent_win_rate = self.calculate_win_rate()
        
        if recent_win_rate > win_rate_threshold:
            return False, f"Recent win rate ({recent_win_rate:.2f}%) above threshold ({win_rate_threshold}%)"
        
        return True, f"Recent win rate ({recent_win_rate:.2f}%) below threshold ({win_rate_threshold}%)"
    
    def train_updated_model(self, test_size=0.2, random_state=42):
        """
        Train an updated model using all available trade data.
        
        Args:
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducible results
            
        Returns:
            dict: Training results including metrics and model path
        """
        logger.info("Starting incremental model training...")
        
        # Load all trade data
        df = self.load_trade_data()
        
        if len(df) < self.min_trades:
            error_msg = f"Insufficient trades ({len(df)}) for training (need {self.min_trades})"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Prepare features and labels
        X, y = self.prepare_features_and_labels(df)
        
        if X is None or y is None:
            error_msg = "Failed to prepare features and labels"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight='balanced'
        )
        
        logger.info("Training Random Forest model...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Model performance - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Save updated model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"trained_model_{timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Also save as the current model (overwrite existing)
        current_model_path = os.path.join(self.models_dir, "trained_model.pkl")
        with open(current_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature names for future use
        features_path = os.path.join(self.models_dir, "expected_features.json")
        with open(features_path, 'w') as f:
            json.dump(X.columns.tolist(), f, indent=2)
        
        # Create training report
        training_report = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "trades_used": len(df),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": accuracy,
            "f1_score": f1,
            "features_used": X.columns.tolist(),
            "model_path": model_path,
            "current_model_path": current_model_path
        }
        
        logger.info("Model training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        
        return training_report
    
    def run_incremental_training(self, force=False, win_rate_threshold=60.0):
        """
        Main method to run incremental training with self-regulation.
        
        Args:
            force (bool): Force retraining regardless of performance
            win_rate_threshold (float): Win rate threshold for auto-retraining
            
        Returns:
            dict: Training results
        """
        try:
            logger.info("Starting incremental training assessment...")
            
            if not force:
                should_retrain, reason = self.should_retrain(win_rate_threshold)
                
                if not should_retrain:
                    logger.info(f"Skipping retraining: {reason}")
                    send_trainer_notification(f"🤖 **Training Skipped**: {reason}")
                    return {"success": True, "skipped": True, "reason": reason}
                
                logger.info(f"Proceeding with retraining: {reason}")
            
            # Perform training
            result = self.train_updated_model()
            
            if result["success"]:
                # Send success notification
                notification_msg = f"""🤖 **Incremental Training Complete!**

📊 **Results:**
• Trades Used: {result['trades_used']}
• Test Accuracy: {result['accuracy']:.3f}
• Test F1 Score: {result['f1_score']:.3f}
• Features: {len(result['features_used'])}

🎯 Model updated with real trading data!"""
                
                send_trainer_notification(notification_msg)
                logger.info("Training notification sent")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during incremental training: {e}"
            logger.error(error_msg)
            send_trainer_notification(f"❌ **Training Failed**: {error_msg}")
            return {"success": False, "error": error_msg}


def main():
    """
    CLI interface for incremental training.
    """
    parser = argparse.ArgumentParser(description="Incremental Model Training")
    parser.add_argument("--model-name", default="random_forest_v1", 
                       help="Model name for training data")
    parser.add_argument("--force", action="store_true", 
                       help="Force retraining regardless of performance")
    parser.add_argument("--win-rate-threshold", type=float, default=60.0,
                       help="Win rate threshold for auto-retraining")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check if retraining is needed")
    
    args = parser.parse_args()
    
    trainer = IncrementalTrainer(args.model_name)
    
    if args.check_only:
        should_retrain, reason = trainer.should_retrain(args.win_rate_threshold)
        print(f"Should retrain: {should_retrain}")
        print(f"Reason: {reason}")
        return
    
    result = trainer.run_incremental_training(args.force, args.win_rate_threshold)
    
    if result["success"]:
        if result.get("skipped"):
            print(f"Training skipped: {result['reason']}")
        else:
            print("Training completed successfully!")
            print(f"Accuracy: {result['accuracy']:.3f}")
            print(f"F1 Score: {result['f1_score']:.3f}")
            print(f"Model saved to: {result['model_path']}")
    else:
        print(f"Training failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

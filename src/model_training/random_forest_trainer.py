import os
import pandas as pd
import numpy as np
import pickle
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import logging
from src.model_training.local_data_loader import fetch_parquet_data_from_local
from src.model_training.common import preprocess_data, save_model, fill_missing_candles
from src.model_training.trainer_diagnostics import TrainerDiagnostics, get_trainer_diagnostics
from src.trading_stats import get_stats_manager, TrainingMetrics
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import RANDOM_FOREST_PARAMS, get_model_path, TRAIN_TEST_SPLIT, RANDOM_STATE
from src.discord_notifications import send_rf_trainer_notification

# --- CONFIG ---
MODEL_OUT_PATH = get_model_path("random_forest", "trained_model.pkl")
TRAIN_SPLIT_RATIO = TRAIN_TEST_SPLIT

RF_PARAMS = RANDOM_FOREST_PARAMS

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Trainer")

class RandomForestTrainer:
    """Random Forest model trainer class"""
    
    def __init__(self, params=None):
        """Initialize the trainer with parameters"""
        self.params = params or RF_PARAMS
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, X):
        """Prepare features for training"""
        return self.scaler.fit_transform(X)
        
    def train(self, X, y):
        """Train the Random Forest model"""
        logger.info("🚂 Training Random Forest model...")
        
        # Prepare features
        X_scaled = self.prepare_features(X)
        
        # Create and train model
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_scaled, y)
        
        logger.info("✅ Random Forest training completed")
        return self.model
        
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'params': self.params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"✅ Model saved to {filepath}")

# --- MAIN ---
def main():
    logger.info("🚂 Starting training pipeline...")
    send_rf_trainer_notification("🚂 **Random Forest Training Started**: Initializing model training pipeline")

    df = fetch_parquet_data_from_local()
    X, y, groups = preprocess_data(df)    # Sanity check before training
    if X.shape[0] < 500 or len(np.unique(y)) < 2:
        logger.error("🚫 Not enough valid data to train. Exiting.")
        send_rf_trainer_notification("🚫 **Training Failed**: Not enough valid data to train Random Forest model")
        return

    send_rf_trainer_notification(f"📊 **Data Prepared**: {X.shape[0]} samples with {X.shape[1]} features ready for Random Forest training")
    
    unique_groups = np.unique(groups)
    n_train = int(TRAIN_SPLIT_RATIO * len(unique_groups))
    train_groups = unique_groups[:n_train]
    test_groups = unique_groups[n_train:]
    
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(**RF_PARAMS))
    ])
    logger.info("🌲 Training Random Forest model...")
    send_rf_trainer_notification("🌲 **Model Training**: Random Forest model training in progress...")
    
    # Record training start time
    training_start_time = time.time()
    
    pipeline.fit(X_train, y_train)
    
    # Record training end time
    training_time = time.time() - training_start_time
    
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    # Probabilities for AUC-ROC
    train_probs = pipeline.predict_proba(X_train)
    test_probs = pipeline.predict_proba(X_test)

    # Metrics for train set
    train_acc = accuracy_score(y_train, train_preds)
    train_precision = precision_score(y_train, train_preds, average='macro')
    train_recall = recall_score(y_train, train_preds, average='macro')
    train_f1 = f1_score(y_train, train_preds, average='macro')
    train_auc = roc_auc_score(y_train, train_probs, multi_class='ovr')

    logger.info(f"📊 Train Accuracy: {train_acc:.4f}")
    logger.info(f"📊 Train Precision: {train_precision:.4f}")
    logger.info(f"📊 Train Recall: {train_recall:.4f}")
    logger.info(f"📊 Train F1 Score: {train_f1:.4f}")
    logger.info(f"📊 Train AUC-ROC: {train_auc:.4f}")
    logger.info(f"🧾 Train Classification Report:\n{classification_report(y_train, train_preds)}")

    # Metrics for test set
    test_acc = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, average='macro')
    test_recall = recall_score(y_test, test_preds, average='macro')
    test_f1 = f1_score(y_test, test_preds, average='macro')
    test_auc = roc_auc_score(y_test, test_probs, multi_class='ovr')

    logger.info(f"📈 Test Accuracy: {test_acc:.4f}")
    logger.info(f"📈 Test Precision: {test_precision:.4f}")
    logger.info(f"📈 Test Recall: {test_recall:.4f}")
    logger.info(f"📈 Test F1 Score: {test_f1:.4f}")
    logger.info(f"📈 Test AUC-ROC: {test_auc:.4f}")
    logger.info(f"🧾 Test Classification Report:\n{classification_report(y_test, test_preds)}")
    
    # Confusion Matrix for Test Set
    test_cm = confusion_matrix(y_test, test_preds)
    logger.info(f"🧾 Test Confusion Matrix:\n{test_cm}")
    
    # Generate comprehensive training diagnostics
    logger.info("📊 Generating training diagnostics...")
    try:
        diagnostics = TrainerDiagnostics("random_forest_v1")
        
        # Create training scores (using accuracy as proxy for loss)
        train_scores = [train_acc] * 5  # Simulate training curve
        val_scores = [test_acc] * 5     # Simulate validation curve
        
        # Get model for diagnostics
        rf_model = pipeline.named_steps['model']
        
        # Analyze training results
        diag_results = diagnostics.analyze_training_results(
            train_scores=train_scores,
            val_scores=val_scores,
            model=rf_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            feature_names=X.columns.tolist(),
            training_time=training_time
        )
        
        # Create training metrics for stats manager
        stats_mgr = get_stats_manager()
        training_metrics = TrainingMetrics(
            model_name="random_forest_v1",
            train_loss=1.0 - train_acc,  # Convert accuracy to loss
            val_loss=1.0 - test_acc,
            overfit_risk=diag_results['overfit_risk'],
            best_features=diag_results['best_features'],
            avg_confidence=diag_results['avg_confidence'],
            winrate_predicted=diag_results['winrate_predicted'],
            winrate_actual=diag_results['winrate_actual'],
            training_time=training_time,
            timestamp=diag_results['timestamp'],
            feature_importance=diag_results['feature_importance'],
            confidence_distribution=diag_results['confidence_distribution']
        )
        
        # Record training metrics
        stats_mgr.record_training_metrics(training_metrics)
        
        # Display training summary
        summary = diagnostics.format_training_summary(diag_results)
        logger.info(f"\n{summary}")
        
        logger.info("✅ Training diagnostics completed and saved")
        
    except Exception as e:
        logger.warning(f"Failed to generate training diagnostics: {e}")
    
    # Save expected features for trade runner compatibility
    expected_features = X.columns.tolist()
    expected_features_path = get_model_path("random_forest", "expected_features.json")
    with open(expected_features_path, "w") as f:
        json.dump(expected_features, f)

    logger.info(f"✅ Expected features saved to {expected_features_path}")
    save_model(pipeline, MODEL_OUT_PATH)

    logger.info("✅ Training pipeline completed.")
    
    # Send comprehensive training completion notification
    notification_msg = f"""🎯 **Random Forest Training Complete!**
    
📊 **Final Results:**
• Test Accuracy: {test_acc:.4f}
• Test F1 Score: {test_f1:.4f}  
• Test AUC-ROC: {test_auc:.4f}
• Model saved to: {MODEL_OUT_PATH}

🌲 The Random Forest is ready for battle!"""
    
    send_rf_trainer_notification(notification_msg)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Training manually interrupted.")
        import sys
        sys.exit(0)




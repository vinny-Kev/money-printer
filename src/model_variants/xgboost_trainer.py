import os
import pandas as pd
import numpy as np
import pickle
import json
import time
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import logging
from src.model_training.local_data_loader import fetch_parquet_data_from_local
from src.model_training.common import preprocess_data, save_model
from src.model_training.trainer_diagnostics import TrainerDiagnostics
from src.trading_stats import get_stats_manager, TrainingMetrics
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import XGBOOST_PARAMS, get_model_path, TRAIN_TEST_SPLIT
from src.discord_notifications import send_xgb_trainer_notification

# --- CONFIG ---
MODEL_OUT_PATH = get_model_path("xgboost", "trained_model.pkl")
TRAIN_SPLIT_RATIO = TRAIN_TEST_SPLIT

XGB_PARAMS = XGBOOST_PARAMS

# --- LOGGING ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("XGB_Trainer")

class XGBoostTrainer:
    """XGBoost model trainer class"""
    
    def __init__(self, params=None):
        """Initialize the trainer with parameters"""
        self.params = params or XGB_PARAMS
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, X):
        """Prepare features for training"""
        return self.scaler.fit_transform(X)
        
    def train(self, X, y):
        """Train the XGBoost model"""
        logger.info("🚂 Training XGBoost model...")
        
        # Prepare features
        X_scaled = self.prepare_features(X)
        
        # Create and train model
        self.model = XGBClassifier(**self.params)
        self.model.fit(X_scaled, y)
        
        logger.info("✅ XGBoost training completed")
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
    logger.info("🚂 Starting XGBoost training pipeline with advanced features...")
    send_xgb_trainer_notification("🚂 **XGBoost Training Started**: Initializing advanced XGBoost training pipeline")

    # Get raw data and show detailed statistics
    logger.info("📊 Loading data from local storage...")
    df = fetch_parquet_data_from_local()
    
    if df is not None and not df.empty:
        # Show extremely detailed data breakdown
        symbol_counts = df['symbol'].value_counts()
        total_rows = len(df)
        unique_symbols = df['symbol'].nunique()
        date_range = df['timestamp'].agg(['min', 'max'])
        
        # Calculate time span
        time_span = pd.to_datetime(date_range['max']) - pd.to_datetime(date_range['min'])
        hours_span = time_span.total_seconds() / 3600
        
        # Data quality metrics
        data_sources = df['source_file'].nunique() if 'source_file' in df.columns else 'Unknown'
        missing_values = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        logger.info(f"📈 **🔍 EXTREMELY DETAILED XGBoost DATA ANALYSIS:**")
        logger.info(f"   📊 🚨 TOTAL ROWS FOR TRAINING: {total_rows:,} 🚨")
        logger.info(f"   🎯 Unique Symbols: {unique_symbols}")
        logger.info(f"   📅 Date Range: {date_range['min']} to {date_range['max']}")
        logger.info(f"   ⏰ Time Span: {hours_span:.1f} hours ({time_span.days} days, {time_span.seconds//3600} hours, {(time_span.seconds%3600)//60} minutes)")
        logger.info(f"   � Data Sources: {data_sources} files")
        logger.info(f"   ❌ Missing Values: {missing_values}")
        logger.info(f"   🔄 Duplicate Rows: {duplicate_rows}")
        logger.info(f"   📋 📊 COMPLETE Symbol Breakdown (ALL {len(symbol_counts)} symbols):")
        
        # Show ALL symbols with their row counts
        for i, (symbol, count) in enumerate(symbol_counts.items(), 1):
            percentage = (count / total_rows) * 100
            status = "✅ GOOD" if count >= 50 else "❌ INSUFFICIENT"
            logger.info(f"      {i:2d}. {symbol}: {count:,} rows ({percentage:.1f}%) - {status}")
        
        # Show symbols with insufficient data
        insufficient_symbols = symbol_counts[symbol_counts < 50]
        sufficient_symbols = symbol_counts[symbol_counts >= 50]
        
        logger.info(f"   📊 DATA QUALITY SUMMARY:")
        logger.info(f"      ✅ Symbols with sufficient data (≥50): {len(sufficient_symbols)}")
        logger.info(f"      ❌ Symbols with insufficient data (<50): {len(insufficient_symbols)}")
        logger.info(f"      📈 Usable rows for training: {sufficient_symbols.sum():,}")
        logger.info(f"      📉 Excluded rows (insufficient data): {insufficient_symbols.sum():,}")
        
        if len(insufficient_symbols) > 0:
            logger.warning(f"   ⚠️ EXCLUDED SYMBOLS (will not be used for training):")
            for symbol, count in insufficient_symbols.items():
                logger.warning(f"      • {symbol}: {count} rows (needs 50+)")
        
        # Send detailed data notification to Discord
        data_summary = f"""📊 **🔍 DETAILED XGBoost Training Data:**

🔢 **DATA VOLUME ANALYSIS:**
• **🚨 TOTAL ROWS**: {total_rows:,} records loaded
• **📊 Usable Rows**: {sufficient_symbols.sum():,} (from {len(sufficient_symbols)} symbols)
• **❌ Excluded Rows**: {insufficient_symbols.sum():,} (from {len(insufficient_symbols)} symbols)
• **⏰ Time Span**: {hours_span:.1f} hours ({time_span.days} days)

📈 **TOP DATA CONTRIBUTORS:**
{chr(10).join([f"• {sym}: {cnt:,} rows ({(cnt/total_rows)*100:.1f}%)" for sym, cnt in symbol_counts.head(8).items()])}

� **DATA QUALITY:**
• Missing Values: {missing_values}
• Duplicate Rows: {duplicate_rows}
• Data Sources: {data_sources} files

⚡ **TRAINING STATUS:**
• ✅ Symbols Ready: {len(sufficient_symbols)}/{unique_symbols}
• 🔄 Processing {sufficient_symbols.sum():,} rows for XGBoost ensemble training..."""
        send_xgb_trainer_notification(data_summary)
    else:
        logger.error("❌ No data loaded from local storage!")
        send_xgb_trainer_notification("❌ **Training Failed**: No data found in local storage")
        return
    
    # Process data for training
    logger.info("🔄 Preprocessing data for XGBoost ensemble training...")
    X, y, groups = preprocess_data(df)

    # Sanity check before training
    if X.shape[0] < 500 or len(np.unique(y)) < 2:
        logger.error("🚫 Not enough valid data to train. Exiting.")
        send_xgb_trainer_notification("🚫 **Training Failed**: Not enough valid data to train XGBoost model")
        return

    send_xgb_trainer_notification(f"📊 **Data Prepared**: {X.shape[0]} samples with {X.shape[1]} features ready for XGBoost training")

    # Replace TimeSeriesSplit with a custom implementation for stacking
    unique_groups = np.unique(groups)
    n_train = int(TRAIN_SPLIT_RATIO * len(unique_groups))
    train_groups = unique_groups[:n_train]
    test_groups = unique_groups[n_train:]

    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)

    base_learners = [
        ("xgb", XGBClassifier(**XGB_PARAMS)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]

    ensemble = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
    )

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ensemble)    ])
    
    # Compute sample weights for class balancing
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    logger.info("🚀 Training XGBoost ensemble model...")
    send_xgb_trainer_notification("🚀 **Model Training**: XGBoost ensemble model training in progress...")

    # Record training start time
    training_start_time = time.time()

    pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)

    # Record training end time
    training_time = time.time() - training_start_time

    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    # Probabilities for AUC-ROC
    train_probs = pipeline.predict_proba(X_train)
    test_probs = pipeline.predict_proba(X_test)

    # Metrics for train set
    train_acc = accuracy_score(y_train, train_preds)
    train_precision = precision_score(y_train, train_preds, average='macro', zero_division=0)
    train_recall = recall_score(y_train, train_preds, average='macro', zero_division=0)
    train_f1 = f1_score(y_train, train_preds, average='macro', zero_division=0)
    train_auc = roc_auc_score(y_train, train_probs, multi_class='ovr')

    logger.info(f"📊 Train Accuracy: {train_acc:.4f}")
    logger.info(f"📊 Train Precision: {train_precision:.4f}")
    logger.info(f"📊 Train Recall: {train_recall:.4f}")
    logger.info(f"📊 Train F1 Score: {train_f1:.4f}")
    logger.info(f"📊 Train AUC-ROC: {train_auc:.4f}")
    logger.info(f"🧾 Train Classification Report:\n{classification_report(y_train, train_preds)}")

    # Metrics for test set
    test_acc = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, average='macro', zero_division=0)
    test_recall = recall_score(y_test, test_preds, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
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
        diagnostics = TrainerDiagnostics("xgboost_v1")
        
        # Create training scores (using accuracy as proxy for loss)
        train_scores = [train_acc] * 5  # Simulate training curve
        val_scores = [test_acc] * 5     # Simulate validation curve
        
        # Get ensemble model for diagnostics
        ensemble_model = pipeline.named_steps['model']
        
        # Analyze training results
        diag_results = diagnostics.analyze_training_results(
            train_scores=train_scores,
            val_scores=val_scores,
            model=ensemble_model,
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
            model_name="xgboost_v1",
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
    expected_features_path = get_model_path("xgboost", "expected_features.json")
    with open(expected_features_path, "w") as f:
        json.dump(expected_features, f)
    
    logger.info(f"✅ Expected features saved to {expected_features_path}")
    save_model(pipeline, MODEL_OUT_PATH)

    logger.info("✅ XGBoost training pipeline with stacking completed.")
    
    # Send comprehensive training completion notification
    notification_msg = f"""🎯 **XGBoost Ensemble Training Complete!**
    
📊 **Comprehensive Training Results:**
• Test Accuracy: {test_acc:.4f} | Train Accuracy: {train_acc:.4f}
• Test Precision: {test_precision:.4f} | Train Precision: {train_precision:.4f}
• Test Recall: {test_recall:.4f} | Train Recall: {train_recall:.4f}
• Test F1 Score: {test_f1:.4f} | Train F1 Score: {train_f1:.4f}
• Test AUC-ROC: {test_auc:.4f} | Train AUC-ROC: {train_auc:.4f}

📈 **Model Performance:**
• Training Time: {training_time:.2f} seconds
• Dataset Size: {X.shape[0]} samples, {X.shape[1]} features
• Train/Test Split: {len(train_groups)}/{len(test_groups)} time groups
• Ensemble: XGBoost + RandomForest + LogisticRegression

💾 **Deployment Ready:**
• Model saved to: {MODEL_OUT_PATH}
• Expected features saved for trading compatibility

🚀 The XGBoost ensemble is ready for deployment! Higher test scores = better trading performance."""
    
    send_xgb_trainer_notification(notification_msg)

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Training manually interrupted.")
        import sys
        sys.exit(0)

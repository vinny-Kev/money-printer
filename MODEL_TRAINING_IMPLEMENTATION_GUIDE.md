# Model Training Implementation Guide

## Overview
This document describes the robust model training implementation in the Money Printer trading bot, specifically focusing on time series split methodology, imbalanced data handling, and hyperparameter configuration.

## Time Series Split Implementation

### Why Time Series Split?
Traditional random splits can lead to **data leakage** in time series data, where future information inadvertently influences past predictions. Our implementation uses a **chronological time-based split** to ensure realistic backtesting.

### Implementation Details

```python
# From random_forest_trainer.py and xgboost_trainer.py
unique_groups = np.unique(groups)  # groups = timestamps/coins
n_train = int(TRAIN_SPLIT_RATIO * len(unique_groups))
train_groups = unique_groups[:n_train]  # Earlier time periods
test_groups = unique_groups[n_train:]   # Later time periods

train_mask = np.isin(groups, train_groups)
test_mask = np.isin(groups, test_groups)
```

### Key Features:
1. **Temporal Ordering**: Training data always comes before test data chronologically
2. **No Data Leakage**: Future information never influences past predictions
3. **Realistic Validation**: Mimics real-world deployment where you only have past data
4. **Group-Based**: Split by time periods/coins rather than individual samples

## Imbalanced Data Handling

### Class Imbalance Problem
Cryptocurrency trading signals often exhibit class imbalance:
- **Hold/Sell signals**: Often more frequent (market uncertainty)
- **Buy signals**: Less frequent but critical for profitability

### Solutions Implemented

#### 1. Random Forest (Balanced Class Weights)
```python
RANDOM_FOREST_PARAMS = {
    "class_weight": "balanced",  # Automatically adjusts for class imbalance
    "n_estimators": 200,
    "max_depth": 12,
    # ... other parameters
}
```

#### 2. XGBoost (Sample Weights)
```python
# Compute balanced sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
```

### Metrics for Imbalanced Data
We use **macro-averaged metrics** which treat all classes equally:

```python
train_precision = precision_score(y_train, train_preds, average='macro')
train_recall = recall_score(y_train, train_preds, average='macro')
train_f1 = f1_score(y_train, train_preds, average='macro')
```

## Comprehensive Metrics Reporting

### Training Metrics Displayed
Both trainers report comprehensive metrics to users via Discord:

#### Core Metrics (Train & Test):
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve

#### Additional Diagnostics:
- **Confusion Matrix**: Detailed breakdown of predictions
- **Classification Report**: Per-class performance
- **Feature Importance**: Most influential indicators
- **Training Time**: Model training duration
- **Sample Count**: Total training samples

### Discord Integration
The Discord bot displays all metrics in real-time:

```
📊 Training Metrics:
• Accuracy: 0.8542
• Precision: 0.8301
• Recall: 0.8234
• F1 Score: 0.8411

📈 Test Metrics:
• Accuracy: 0.7823
• Precision: 0.7645
• Recall: 0.7534
• F1 Score: 0.7712

🎯 Performance Summary:
• AUC-ROC: 0.8234
• Training Time: 32.4s
• Data Split: Time series balanced split
• Samples: 15,432
```

## Hyperparameter Configuration & Tuning

### Random Forest Parameters
Configured in `src/config.py`:

```python
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,        # Number of trees
    "max_depth": 12,           # Maximum tree depth
    "min_samples_leaf": 4,     # Minimum samples per leaf
    "min_samples_split": 8,    # Minimum samples to split
    "max_features": "sqrt",    # Features per split
    "class_weight": "balanced", # Handle imbalance
    "random_state": 42,        # Reproducibility
    "n_jobs": -1,             # Use all CPU cores
    "oob_score": True,        # Out-of-bag validation
    "bootstrap": True,        # Bootstrap sampling
}
```

### XGBoost Parameters
```python
XGBOOST_PARAMS = {
    "n_estimators": 300,        # Number of boosting rounds
    "max_depth": 10,           # Maximum tree depth
    "learning_rate": 0.05,     # Step size shrinkage
    "subsample": 0.8,          # Row sampling ratio
    "colsample_bytree": 0.8,   # Column sampling ratio
    "objective": "binary:logistic", # Loss function
    "random_state": 42,        # Reproducibility
    "eval_metric": "logloss"   # Evaluation metric
}
```

### Hyperparameter Tuning Options

#### Manual Tuning
Modify parameters in `src/config.py` and retrain:

```bash
# Edit src/config.py
# Change RANDOM_FOREST_PARAMS or XGBOOST_PARAMS
# Then retrain via Discord: /train_model random_forest
```

#### Advanced Tuning (Future Enhancement)
For automated hyperparameter optimization, consider:

1. **Grid Search**: Systematic parameter exploration
2. **Random Search**: Random parameter sampling  
3. **Bayesian Optimization**: Smart parameter search
4. **Optuna**: Modern hyperparameter optimization

### Model Selection Guidance

#### When to Use Random Forest:
- **Interpretability**: Feature importance is crucial
- **Stability**: Robust to outliers and noise
- **Speed**: Faster training for quick iterations
- **Conservative**: Lower risk, steady performance

#### When to Use XGBoost:
- **Performance**: Often achieves higher accuracy
- **Flexibility**: More tuning options available
- **Ensemble**: Uses stacking with Random Forest
- **Competition**: State-of-the-art gradient boosting

## Validation & Safety

### Model Validation Pipeline
1. **Data Quality Checks**: Ensure sufficient samples and class diversity
2. **Time Series Split**: Proper temporal validation
3. **Imbalance Handling**: Balanced metrics and weights
4. **Performance Monitoring**: Comprehensive metric tracking
5. **Overfitting Detection**: Train vs validation performance gaps

### Safety Mechanisms
- **Emergency Stop**: Manual trading halt via Discord
- **Paper Trading**: Test models without real money
- **Conservative Defaults**: Balanced, robust parameters
- **Comprehensive Logging**: Full audit trail of decisions
- **Multi-Model Validation**: Random Forest + XGBoost consensus

## Best Practices

### Data Preparation
1. **Clean Data**: Remove outliers and invalid entries
2. **Feature Engineering**: Technical indicators with proper periods
3. **Normalization**: StandardScaler for consistent scaling
4. **Time Alignment**: Ensure proper temporal ordering

### Model Training
1. **Sufficient Data**: Minimum 500 samples recommended
2. **Class Balance**: Monitor class distribution
3. **Validation**: Always use time series split
4. **Metric Focus**: Prioritize F1 and AUC for trading

### Deployment
1. **Backtesting**: Validate on historical data
2. **Paper Trading**: Test with fake money first
3. **Small Stakes**: Start with minimal position sizes
4. **Monitoring**: Continuous performance tracking

---

*This implementation ensures robust, production-ready model training with proper time series methodology, imbalance handling, and comprehensive metrics reporting for informed trading decisions.*

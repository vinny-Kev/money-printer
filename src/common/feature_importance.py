import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json

# --- Set base path relative to script ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/models"))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "trained_model_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "important_features.json")

def load_artifacts(model_path, scaler_path, features_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(features_path, 'r') as f:
        features = json.load(f)
    return model, scaler, features

def plot_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        series = pd.Series(importances, index=features).sort_values(ascending=True)
        plt.figure(figsize=(8, 6))
        series.plot(kind='barh', color='skyblue')
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.grid(axis='x')
        plt.show()
    else:
        print("Model does not support feature_importances_ attribute.")

if __name__ == "__main__":
    model, scaler, features = load_artifacts(MODEL_PATH, SCALER_PATH, FEATURES_PATH)
    plot_feature_importance(model, features)

import pandas as pd
import pickle
import sys
import os

# === Config ===
MODEL_PATH = "trained_model.pkl"
INPUT_CSV = "btc_model_data.csv"  # Change or use CLI

# === Load Model ===
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# === Load Input CSV ===
if not os.path.exists(INPUT_CSV):
    print(f"Input CSV not found: {INPUT_CSV}")
    sys.exit(1)

try:
    df = pd.read_csv(INPUT_CSV)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# === Predict ===
try:
    predictions = model.predict(df)
    print("Predictions:")
    for i, p in enumerate(predictions):
        print(f"Sample {i+1}: {p}")
except Exception as e:
    print(f"Error during prediction: {e}")

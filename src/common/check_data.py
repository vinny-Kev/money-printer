import pandas as pd
import os

DATA_DIR = r"Z:\money_printer\data\scraped_data"
for fname in os.listdir(DATA_DIR):
    if fname.endswith("_model_data.parquet"):
        fpath = os.path.join(DATA_DIR, fname)
        df = pd.read_parquet(fpath)
        print(f"{fname}: {df.shape[0]} rows, columns: {list(df.columns)}")
        print(df[["rsi", "macd"]].isnull().sum())
        print("-" * 40)
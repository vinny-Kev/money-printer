# Test script to check module imports in Docker
try:
    import src.trading_bot
    import src.data_collector
    import src.model_training
    print("All modules imported successfully!")
except Exception as e:
    print(f"Import error: {e}")

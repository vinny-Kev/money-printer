#!/usr/bin/env python3
"""
Quick test for ensemble trainer
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing imports...")

try:
    from src.storage.enhanced_storage_manager import EnhancedStorageManager
    print("✅ Enhanced Storage Manager imported")
except Exception as e:
    print(f"❌ Enhanced Storage Manager failed: {e}")

try:
    from src.config import MODELS_DIR, RANDOM_STATE, DATA_ROOT
    print("✅ Config imported")
    print(f"   MODELS_DIR: {MODELS_DIR}")
    print(f"   DATA_ROOT: {DATA_ROOT}")
except Exception as e:
    print(f"❌ Config failed: {e}")

try:
    from src.model_training.ensemble_production_trainer import EnsembleProductionTrainer
    print("✅ Ensemble Production Trainer imported")
except Exception as e:
    print(f"❌ Ensemble Production Trainer failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting basic functionality...")

try:
    # Create storage manager
    storage_manager = EnhancedStorageManager(
        local_backup_dir=str(DATA_ROOT / "scraped_data" / "parquet_files")
    )
    print("✅ Storage manager created")
    
    # Create trainer
    trainer = EnsembleProductionTrainer(
        storage_manager=storage_manager,
        min_rows_total=500,  # Lower for testing
        min_time_span_hours=1,  # Lower for testing
    )
    print("✅ Trainer created")
    
    # Just test data loading
    print("Testing data loading...")
    data = trainer.storage_manager.load_combined_data()
    print(f"✅ Data loaded: {data.shape[0]} rows")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!")

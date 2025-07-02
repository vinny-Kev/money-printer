#!/usr/bin/env python3
"""
Simple Local Model Training Runner
Trains the ensemble models using local data only
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

if __name__ == "__main__":
    print("🤖 Starting Local Ensemble Model Training...")
    print("=" * 50)
    
    # Import and run the trainer
    try:
        from src.model_training.ensemble_production_trainer import main
        print("✅ Imports successful, starting training...")
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔍 Trying alternative import method...")
        try:
            # Direct execution method
            import subprocess
            trainer_path = project_root / "src" / "model_training" / "ensemble_production_trainer.py"
            print(f"✅ Running trainer directly: {trainer_path}")
            subprocess.run([sys.executable, str(trainer_path)], check=True)
        except Exception as e2:
            print(f"❌ Alternative import also failed: {e2}")
            print("\n💡 Debug information:")
            print(f"   Current working directory: {Path.cwd()}")
            print(f"   Script location: {__file__}")
            print(f"   Python path: {sys.path[:3]}...")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

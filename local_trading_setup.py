#!/usr/bin/env python3
"""
🚀 MONEY PRINTER - LOCAL TRADING SETUP
Complete local setup for crypto trading system - NO RAILWAY/DOCKER NEEDED!
This script sets up everything needed to run the trading system locally.
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

class LocalTradingSetup:
    def __init__(self):
        self.root_dir = Path.cwd()
        self.data_dir = self.root_dir / "data"
        self.models_dir = self.data_dir / "models"
        self.scraped_data_dir = self.data_dir / "scraped_data" / "parquet_files"
        self.logs_dir = self.root_dir / "logs"
        self.src_dir = self.root_dir / "src"
        
        # Setup status
        self.setup_status = {
            'directories_created': False,
            'packages_installed': False,
            'environment_configured': False,
            'data_available': False,
            'models_available': False,
            'config_validated': False,
            'ready_for_trading': False
        }
    
    def print_header(self):
        """Print setup header"""
        print("=" * 70)
        print("🚀 MONEY PRINTER - LOCAL TRADING SYSTEM SETUP")
        print("=" * 70)
        print("🎯 Setting up complete crypto trading system locally")
        print("💡 No Docker, No Railway - Pure local Python setup!")
        print(f"📍 Working directory: {self.root_dir}")
        print("=" * 70)
    
    def create_directories(self):
        """Create all required directories"""
        print("\n📁 CREATING DIRECTORY STRUCTURE")
        print("-" * 40)
        
        directories = [
            self.data_dir,
            self.models_dir,
            self.scraped_data_dir,
            self.logs_dir,
            self.data_dir / "backups",
            self.data_dir / "exports",
            self.models_dir / "ensemble",
            self.models_dir / "individual",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory.relative_to(self.root_dir)}")
        
        self.setup_status['directories_created'] = True
        print("✅ All directories created successfully!")
    
    def install_packages(self):
        """Install all required Python packages"""
        print("\n📦 INSTALLING REQUIRED PACKAGES")
        print("-" * 40)
        
        # Core packages
        core_packages = [
            "python-binance>=1.0.19",
            "discord.py>=2.3.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
            "python-dotenv>=1.0.0",
            "pyarrow>=12.0.0",
            "requests>=2.31.0",
            "aiohttp>=3.8.0",
            "asyncio-throttle>=1.0.2",
        ]
        
        # Optional ML packages
        optional_packages = [
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "ta>=0.10.0",  # Technical analysis
            "plotly>=5.0.0",  # Visualization
        ]
        
        print("Installing core packages...")
        failed_packages = []
        
        for package in core_packages:
            try:
                print(f"   📦 Installing {package}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, check=True, timeout=300)
                print(f"   ✅ {package} installed")
            except Exception as e:
                print(f"   ❌ Failed to install {package}: {e}")
                failed_packages.append(package)
        
        print("\nInstalling optional packages...")
        for package in optional_packages:
            try:
                print(f"   📦 Installing {package}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, check=True, timeout=300)
                print(f"   ✅ {package} installed")
            except Exception as e:
                print(f"   ⚠️ Optional package {package} failed: {e}")
        
        if failed_packages:
            print(f"\n⚠️ Some packages failed to install: {failed_packages}")
            print("💡 The system may still work, but some features might be limited")
        else:
            print("\n✅ All packages installed successfully!")
        
        self.setup_status['packages_installed'] = len(failed_packages) == 0
    
    def setup_environment(self):
        """Setup environment configuration"""
        print("\n⚙️ SETTING UP ENVIRONMENT")
        print("-" * 40)
        
        # Create .env file template if it doesn't exist
        env_file = self.root_dir / ".env"
        if not env_file.exists():
            env_template = """# MONEY PRINTER - LOCAL ENVIRONMENT CONFIGURATION
# Copy this file and fill in your actual API keys

# Binance API Configuration (for trading)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_secret_key_here
BINANCE_TESTNET=true

# Discord Bot Configuration (optional)
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CHANNEL_ID=your_discord_channel_id_here

# Trading Configuration
TRADING_MODE=paper  # paper or live
DEFAULT_TRADE_AMOUNT=10.0
MAX_POSITIONS=5

# Data Collection Configuration
DATA_COLLECTION_INTERVAL=1m
SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT,LINKUSDT

# Local Storage Configuration
LOCAL_STORAGE_PATH=data/scraped_data/parquet_files
MODEL_STORAGE_PATH=data/models
BACKUP_ENABLED=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_TO_FILE=true
"""
            with open(env_file, 'w') as f:
                f.write(env_template)
            print(f"✅ Created .env template: {env_file}")
            print("💡 Please edit .env file with your actual API keys")
        else:
            print("✅ .env file already exists")
        
        # Test environment loading
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Environment loading test passed")
        except Exception as e:
            print(f"❌ Environment loading failed: {e}")
        
        self.setup_status['environment_configured'] = True
    
    def validate_imports(self):
        """Validate that all core imports work"""
        print("\n🔧 VALIDATING CORE IMPORTS")
        print("-" * 40)
        
        # Add src to path
        sys.path.insert(0, str(self.src_dir))
        
        import_tests = [
            ("pandas", "import pandas as pd"),
            ("numpy", "import numpy as np"),
            ("sklearn", "from sklearn.ensemble import RandomForestClassifier"),
            ("joblib", "import joblib"),
            ("binance", "from binance import Client"),
            ("discord", "import discord"),
            ("dotenv", "from dotenv import load_dotenv"),
        ]
        
        failed_imports = []
        for name, import_code in import_tests:
            try:
                exec(import_code)
                print(f"✅ {name} import successful")
            except Exception as e:
                print(f"❌ {name} import failed: {e}")
                failed_imports.append(name)
        
        if failed_imports:
            print(f"\n⚠️ Some imports failed: {failed_imports}")
            return False
        else:
            print("\n✅ All core imports successful!")
            return True
    
    def check_data_availability(self):
        """Check if training data is available"""
        print("\n📊 CHECKING DATA AVAILABILITY")
        print("-" * 40)
        
        parquet_files = list(self.scraped_data_dir.glob("*.parquet"))
        
        if parquet_files:
            total_size = sum(f.stat().st_size for f in parquet_files)
            size_mb = total_size / (1024 * 1024)
            print(f"✅ Found {len(parquet_files)} data files ({size_mb:.1f} MB)")
            
            # Check file ages
            from datetime import datetime, timedelta
            recent_files = [f for f in parquet_files if 
                          datetime.fromtimestamp(f.stat().st_mtime) > 
                          datetime.now() - timedelta(days=7)]
            
            print(f"📅 Recent files (last 7 days): {len(recent_files)}")
            self.setup_status['data_available'] = len(recent_files) > 0
            
            if not recent_files:
                print("⚠️ No recent data found - consider running data collection")
        else:
            print("❌ No data files found")
            print("💡 Run 'python main.py collect' to gather training data")
            self.setup_status['data_available'] = False
    
    def check_models(self):
        """Check if trained models are available"""
        print("\n🤖 CHECKING MODEL AVAILABILITY")
        print("-" * 40)
        
        model_extensions = ['.joblib', '.pkl', '.json']
        model_files = []
        
        for ext in model_extensions:
            model_files.extend(list(self.models_dir.glob(f"**/*{ext}")))
        
        if model_files:
            print(f"✅ Found {len(model_files)} model files")
            for model_file in model_files[:5]:  # Show first 5
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"   📁 {model_file.name} ({size_mb:.1f} MB)")
            
            if len(model_files) > 5:
                print(f"   ... and {len(model_files) - 5} more")
            
            self.setup_status['models_available'] = True
        else:
            print("❌ No trained models found")
            print("💡 Run 'python src/model_training/ensemble_production_trainer.py' to train models")
            self.setup_status['models_available'] = False
    
    def create_local_runner_scripts(self):
        """Create convenient runner scripts for local use"""
        print("\n📝 CREATING LOCAL RUNNER SCRIPTS")
        print("-" * 40)
        
        # Data collection script
        collector_script = self.root_dir / "run_data_collection.py"
        collector_content = '''#!/usr/bin/env python3
"""Local Data Collection Runner"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("🚀 Starting Local Data Collection...")
    from data_collector.data_scraper import main
    main()
'''
        with open(collector_script, 'w') as f:
            f.write(collector_content)
        print("✅ Created run_data_collection.py")
        
        # Model training script
        trainer_script = self.root_dir / "run_model_training.py"
        trainer_content = '''#!/usr/bin/env python3
"""Local Model Training Runner"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("🚀 Starting Local Model Training...")
    from model_training.ensemble_production_trainer import main
    main()
'''
        with open(trainer_script, 'w') as f:
            f.write(trainer_content)
        print("✅ Created run_model_training.py")
        
        # Trading script
        trading_script = self.root_dir / "run_trading.py"
        trading_content = '''#!/usr/bin/env python3
"""Local Trading Runner"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("🚀 Starting Local Trading...")
    import os
    os.environ['TRADING_MODE'] = 'local'
    from trading.main_trader import main
    main()
'''
        with open(trading_script, 'w') as f:
            f.write(trading_content)
        print("✅ Created run_trading.py")
    
    def remove_railway_files(self):
        """Remove Railway deployment files to avoid confusion"""
        print("\n🗑️ REMOVING RAILWAY DEPLOYMENT FILES")
        print("-" * 40)
        
        railway_files = [
            "railway.json",
            "railway.toml", 
            "Procfile",
            "Dockerfile.railway",
            "railway_health_server.py",
            "src/railway_watchdog.py",
            "src/data_collector/railway_data_scraper.py",
            "test_railway_setup.py",
            "debug_railway.py",
            "analyze_railway_deployment.py",
        ]
        
        removed_files = []
        for file_path in railway_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                try:
                    full_path.unlink()
                    removed_files.append(file_path)
                    print(f"🗑️ Removed: {file_path}")
                except Exception as e:
                    print(f"⚠️ Could not remove {file_path}: {e}")
        
        if removed_files:
            print(f"✅ Removed {len(removed_files)} Railway files")
        else:
            print("✅ No Railway files to remove")
    
    def generate_status_report(self):
        """Generate a comprehensive status report"""
        print("\n📋 SETUP STATUS REPORT")
        print("=" * 50)
        
        for check, status in self.setup_status.items():
            icon = "✅" if status else "❌"
            print(f"{icon} {check.replace('_', ' ').title()}: {'PASSED' if status else 'FAILED'}")
        
        # Overall readiness
        critical_checks = ['directories_created', 'packages_installed', 'environment_configured']
        critical_passed = all(self.setup_status[check] for check in critical_checks)
        
        optional_checks = ['data_available', 'models_available']
        optional_passed = any(self.setup_status[check] for check in optional_checks)
        
        print("\n📊 READINESS ASSESSMENT:")
        if critical_passed:
            print("✅ CRITICAL SETUP: READY")
            if optional_passed:
                print("✅ OPTIONAL COMPONENTS: AVAILABLE")
                print("🎉 SYSTEM FULLY READY FOR TRADING!")
                self.setup_status['ready_for_trading'] = True
            else:
                print("⚠️ OPTIONAL COMPONENTS: MISSING")
                print("💡 You can start data collection and model training")
        else:
            print("❌ CRITICAL SETUP: INCOMPLETE")
            print("🛑 Please fix critical issues before proceeding")
    
    def print_usage_guide(self):
        """Print comprehensive usage guide"""
        print("\n🎮 LOCAL TRADING SYSTEM - USAGE GUIDE")
        print("=" * 60)
        
        print("\n📊 DATA COLLECTION:")
        print("   python run_data_collection.py")
        print("   python main.py collect --hours 2")
        
        print("\n🤖 MODEL TRAINING:")
        print("   python run_model_training.py")
        print("   python src/model_training/ensemble_production_trainer.py")
        
        print("\n💰 TRADING:")
        print("   python run_trading.py")
        print("   python main.py trade")
        
        print("\n🔍 SYSTEM STATUS:")
        print("   python main.py status")
        print("   python check_model_status.py")
        
        print("\n⚙️ CONFIGURATION:")
        print("   Edit .env file for API keys and settings")
        print("   All data stored locally in data/ directory")
        
        print("\n🚀 QUICK START SEQUENCE:")
        print("   1. Edit .env file with your API keys")
        print("   2. python run_data_collection.py  # Collect training data")
        print("   3. python run_model_training.py   # Train models")
        print("   4. python run_trading.py          # Start trading")
        
        print("\n💡 NO DOCKER, NO RAILWAY - PURE LOCAL PYTHON!")
    
    def run_full_setup(self):
        """Run the complete setup process"""
        self.print_header()
        
        try:
            self.create_directories()
            self.install_packages()
            self.setup_environment()
            
            # Validate setup
            imports_ok = self.validate_imports()
            self.setup_status['config_validated'] = imports_ok
            
            self.check_data_availability()
            self.check_models()
            self.create_local_runner_scripts()
            self.remove_railway_files()
            
            self.generate_status_report()
            self.print_usage_guide()
            
        except Exception as e:
            print(f"\n❌ Setup failed with error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main setup function"""
    setup = LocalTradingSetup()
    setup.run_full_setup()

if __name__ == "__main__":
    main()

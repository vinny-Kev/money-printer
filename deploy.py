"""
Deployment script for Money Printer trading system.
Sets up the environment and runs basic health checks.
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_environment_file():
    """Check if .env file exists."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please create a .env file with your API keys:")
        print("BINANCE_API_KEY=your_key")
        print("BINANCE_SECRET_KEY=your_secret")
        return False
    print("✅ .env file found")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_clean.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    directories = [
        "data/scraped_data/parquet_files",
        "data/models/random_forest",
        "data/models/xgboost", 
        "logs",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True

def run_health_check():
    """Run a basic health check."""
    print("🔍 Running health check...")
    try:
        # Test config import
        sys.path.append("src")
        from config import validate_environment
        validate_environment()
        print("✅ Configuration validated")
          # Test data collector import
        from data_collector.local_storage import save_parquet_file
        print("✅ Data collector modules working")
        
        # Test Discord notifications
        from discord_notifications import send_general_notification
        print("✅ Discord notifications working")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Main deployment function."""
    print("🚀 Money Printer Deployment Script")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_environment_file),
        ("Dependencies", install_dependencies),
        ("Directories", setup_directories),
        ("Health Check", run_health_check)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        print(f"\n🔄 {check_name}...")
        if not check_func():
            failed_checks.append(check_name)
    
    print("\n" + "=" * 40)
    if failed_checks:
        print(f"❌ Deployment failed. Issues with: {', '.join(failed_checks)}")
        print("\nPlease fix the issues above and run the script again.")
        return False
    else:
        print("✅ Deployment successful!")
        print("\nNext steps:")
        print("1. Start data collection: python main.py collect")
        print("2. Train models: python main.py train")
        print("3. Start trading: python main.py trade")
        print("4. Check status: python main.py status")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

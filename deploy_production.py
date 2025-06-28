"""
Production Deployment Script
Launches the Money Printer system in production mode
"""
import os
import sys
import logging
import signal
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'production_deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def start_production_deployment():
    """Start the production deployment"""
    
    logger.info("🚀 STARTING MONEY PRINTER PRODUCTION DEPLOYMENT")
    logger.info("=" * 60)
    logger.info(f"📅 Deployment Time: {datetime.now().isoformat()}")
    
    # Verify environment one more time
    logger.info("🔍 Final environment verification...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check critical environment variables
    required_vars = {
        'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY'),
        'BINANCE_SECRET_KEY': os.getenv('BINANCE_SECRET_KEY'),
        'GOOGLE_DRIVE_FOLDER_ID': os.getenv('GOOGLE_DRIVE_FOLDER_ID'),
        'USE_GOOGLE_DRIVE': os.getenv('USE_GOOGLE_DRIVE')
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    logger.info("✅ Environment verification passed")
    
    # Test Google Drive connection
    logger.info("☁️ Testing Google Drive connection...")
    try:
        from drive_manager import EnhancedDriveManager
        dm = EnhancedDriveManager()
        files = dm.list_files_in_folder()
        logger.info(f"✅ Google Drive connected ({len(files)} files)")
    except Exception as e:
        logger.error(f"❌ Google Drive connection failed: {e}")
        return False
    
    # Test Binance connection
    logger.info("🔗 Testing Binance connection...")
    try:
        from binance_wrapper import EnhancedBinanceClient
        from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_TESTNET
        
        client = EnhancedBinanceClient(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_SECRET_KEY,
            testnet=BINANCE_TESTNET
        )
        
        if client.test_connection():
            logger.info("✅ Binance connection successful")
            
            # Get account info for verification
            account_info = client.get_account_info()
            if account_info:
                logger.info(f"✅ Account verified: {account_info['account_type']}")
            
        else:
            logger.error("❌ Binance connection failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Binance connection error: {e}")
        return False
    
    logger.info("🎯 All systems verified - proceeding with deployment")
    return True

def launch_data_collection():
    """Launch the data collection system"""
    
    logger.info("📊 LAUNCHING DATA COLLECTION SYSTEM")
    logger.info("=" * 40)
    
    # Import main production module
    import subprocess
    import threading
    
    # Start data collection in continuous mode
    logger.info("🚀 Starting continuous data collection...")
    
    try:
        # Use subprocess to run the main production script
        cmd = [sys.executable, "main_production.py", "collect"]
        
        logger.info(f"📋 Command: {' '.join(cmd)}")
        logger.info("⏱️ Data collection will run continuously until stopped")
        logger.info("🛑 Use Ctrl+C to stop gracefully")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor the output
        logger.info("📈 Monitoring data collection output...")
        
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Forward the output to our logger
                    print(output.strip())
                    
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal")
            process.terminate()
            process.wait()
            logger.info("✅ Data collection stopped gracefully")
            
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        return False
    
    return True

def main():
    """Main deployment function"""
    
    print("🚀 MONEY PRINTER - PRODUCTION DEPLOYMENT")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Environment verification
    if not start_production_deployment():
        print("❌ Deployment failed - environment verification failed")
        return False
    
    print("\n" + "="*50)
    print("🎉 DEPLOYMENT VERIFICATION COMPLETE")
    print("="*50)
    print("✅ All systems verified and ready")
    print("📊 Data collection system ready to launch")
    print("☁️ Google Drive integration active")
    print("🔒 Security measures in place")
    print("📈 Monitoring and logging enabled")
    print()
    
    # Ask for confirmation
    try:
        confirmation = input("🚀 Start production data collection? (y/N): ")
        if confirmation.lower() in ['y', 'yes']:
            print("\n🎯 Launching production system...")
            return launch_data_collection()
        else:
            print("⏸️ Deployment paused - system ready when you are")
            print("💡 To start later, run: python main_production.py collect")
            return True
            
    except KeyboardInterrupt:
        print("\n⏸️ Deployment cancelled by user")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

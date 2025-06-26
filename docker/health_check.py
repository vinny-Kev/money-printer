#!/usr/bin/env python3
"""
Docker Health Check Endpoint
Simple health check for Docker and Railway deployment
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from src.config import DATA_ROOT, LOGS_DIR
    from src.drive_manager import get_drive_manager
    
    def health_check():
        """Perform health check"""
        checks = {
            "directories": False,
            "drive_manager": False,
            "environment": False
        }
        
        # Check directories
        try:
            required_dirs = [DATA_ROOT, LOGS_DIR]
            checks["directories"] = all(d.exists() for d in required_dirs)
        except:
            pass
        
        # Check environment variables
        try:
            required_env = ["DISCORD_BOT_TOKEN", "BINANCE_API_KEY"]
            checks["environment"] = all(os.getenv(var) for var in required_env)
        except:
            pass
        
        # Check Drive manager (if enabled)
        try:
            if os.getenv("USE_GOOGLE_DRIVE", "false").lower() == "true":
                manager = get_drive_manager()
                checks["drive_manager"] = manager.authenticated
            else:
                checks["drive_manager"] = True  # Not required
        except:
            checks["drive_manager"] = True  # Don't fail on Drive issues
        
        return checks
    
    if __name__ == "__main__":
        checks = health_check()
        
        if all(checks.values()):
            print("✅ Health check passed")
            exit(0)
        else:
            print(f"❌ Health check failed: {checks}")
            exit(1)
            
except Exception as e:
    print(f"❌ Health check error: {e}")
    exit(1)

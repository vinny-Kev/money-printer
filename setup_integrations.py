#!/usr/bin/env python3
"""
Google Drive Setup Script
Helps set up Google Drive integration for the trading bot.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import SECRETS_DIR, DRIVE_CREDENTIALS_PATH, DRIVE_TOKEN_PATH

def setup_google_drive():
    """Setup Google Drive integration"""
    print("🔧 Google Drive Integration Setup")
    print("=" * 50)
    
    # Ensure secrets directory exists
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n📋 Steps to set up Google Drive integration:")
    print("\n1. Go to Google Cloud Console (https://console.cloud.google.com/)")
    print("2. Create a new project or select existing one")
    print("3. Enable the Google Drive API")
    print("4. Create credentials (OAuth 2.0 Client ID)")
    print("5. Download the credentials JSON file")
    print(f"6. Save it as: {DRIVE_CREDENTIALS_PATH}")
    
    print(f"\n📁 Expected file location: {DRIVE_CREDENTIALS_PATH}")
    
    if DRIVE_CREDENTIALS_PATH.exists():
        print("✅ credentials.json found!")
        
        # Validate credentials file
        try:
            with open(DRIVE_CREDENTIALS_PATH, 'r') as f:
                creds = json.load(f)
                
            if 'installed' in creds and 'client_id' in creds['installed']:
                print("✅ Credentials file appears valid")
                
                # Test authentication
                print("\n🔐 Testing authentication...")
                from src.drive_uploader import DriveUploader
                
                uploader = DriveUploader()
                if uploader.test_connection():
                    print("✅ Google Drive connection successful!")
                    print("\n📝 Next steps:")
                    print("1. Set USE_GOOGLE_DRIVE=true in your .env file")
                    print("2. Set GOOGLE_DRIVE_FOLDER_ID in your .env file")
                    print("3. Restart the bot to enable sync")
                else:
                    print("❌ Google Drive connection failed")
                    print("Please check your credentials and try again")
                    
            else:
                print("❌ Invalid credentials file format")
                
        except Exception as e:
            print(f"❌ Error validating credentials: {e}")
            
    else:
        print("❌ credentials.json not found")
        print(f"\nPlease download your credentials and save to: {DRIVE_CREDENTIALS_PATH}")
    
    print(f"\n📂 Create a Google Drive folder for your trading data")
    print("Then get the folder ID from the URL:")
    print("https://drive.google.com/drive/folders/[FOLDER_ID_HERE]")
    
    print(f"\n⚙️  Add these to your .env file:")
    print(f"USE_GOOGLE_DRIVE=true")
    print(f"GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here")

def create_sample_env():
    """Create sample .env entries"""
    env_sample = """
# Railway Configuration
RAILWAY_API_TOKEN=your_railway_api_token_here
RAILWAY_PROJECT_ID=your_railway_project_id_here
RAILWAY_MAX_USAGE_HOURS=450
RAILWAY_WARNING_HOURS=400
RAILWAY_CHECK_INTERVAL=5

# Google Drive Configuration
USE_GOOGLE_DRIVE=false
GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id_here
"""
    
    print("\n📝 Sample .env entries:")
    print(env_sample)
    
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file exists at {env_file}")
    else:
        print(f"❌ .env file not found. Create one at {env_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Google Drive Setup")
    parser.add_argument("--setup", action="store_true", help="Run Google Drive setup")
    parser.add_argument("--env", action="store_true", help="Show sample .env entries")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_google_drive()
    elif args.env:
        create_sample_env()
    else:
        setup_google_drive()
        create_sample_env()

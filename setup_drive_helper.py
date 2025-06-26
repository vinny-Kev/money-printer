#!/usr/bin/env python3
"""
Quick Setup Helper for Enhanced Drive Integration
"""

import os
import json
from pathlib import Path

def main():
    print("🔧 Enhanced Google Drive Setup Helper")
    print("=" * 50)
    
    secrets_dir = Path("secrets")
    service_account_path = secrets_dir / "service_account.json"
    
    print(f"\n📁 Checking secrets directory: {secrets_dir.absolute()}")
    
    if not secrets_dir.exists():
        secrets_dir.mkdir(parents=True)
        print("✅ Created secrets directory")
    
    print(f"\n📋 Service Account Key Location:")
    print(f"   Expected: {service_account_path.absolute()}")
    
    if service_account_path.exists():
        print("✅ Service account key found!")
        
        # Validate the JSON
        try:
            with open(service_account_path, 'r') as f:
                key_data = json.load(f)
            
            if 'client_email' in key_data:
                print(f"📧 Service account email: {key_data['client_email']}")
                print("\n📂 Share your Google Drive folder with this email!")
            else:
                print("❌ Invalid service account key format")
                
        except Exception as e:
            print(f"❌ Error reading service account key: {e}")
    else:
        print("❌ Service account key not found")
        print(f"\n💡 To fix this:")
        print(f"   1. Download your service account JSON key from Google Cloud Console")
        print(f"   2. Save it as: {service_account_path.absolute()}")
    
    # Check .env configuration
    print(f"\n⚙️ Environment Configuration:")
    
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            env_content = f.read()
        
        if 'USE_GOOGLE_DRIVE=true' in env_content:
            print("✅ USE_GOOGLE_DRIVE=true found")
        else:
            print("❌ Add USE_GOOGLE_DRIVE=true to .env")
        
        if 'GOOGLE_DRIVE_FOLDER_ID=' in env_content:
            print("✅ GOOGLE_DRIVE_FOLDER_ID found")
        else:
            print("❌ Add GOOGLE_DRIVE_FOLDER_ID=your_folder_id to .env")
    else:
        print("❌ .env file not found")
        print("💡 Create .env file with:")
        print("   USE_GOOGLE_DRIVE=true")
        print("   GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here")
    
    print(f"\n🧪 Test when ready:")
    print(f"   python src/drive_manager.py --status")
    print(f"   python test_enhanced_integration.py")

if __name__ == "__main__":
    main()

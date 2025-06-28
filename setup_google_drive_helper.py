"""
Google Drive Setup Helper for Money Printer
Provides step-by-step instructions for setting up Google Drive integration
"""
import os
import json
from pathlib import Path

def setup_google_drive():
    """Interactive Google Drive setup"""
    print("🚀 Money Printer - Google Drive Setup")
    print("=" * 50)
    
    # Check if secrets directory exists
    secrets_dir = Path("secrets")
    secrets_dir.mkdir(exist_ok=True)
    
    service_account_path = secrets_dir / "service_account.json"
    
    if service_account_path.exists():
        print("✅ Service account key already exists!")
        
        # Validate the key
        try:
            with open(service_account_path, 'r') as f:
                key_data = json.load(f)
            
            required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in key_data]
            
            if not missing_fields:
                print(f"✅ Service account key is valid")
                print(f"📧 Service account email: {key_data['client_email']}")
                print(f"🗂️ Project ID: {key_data['project_id']}")
                
                # Check environment variables
                folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
                if folder_id:
                    print(f"✅ Google Drive folder ID configured: {folder_id}")
                    print("\n🎉 Google Drive is ready to use!")
                    return True
                else:
                    print("⚠️ GOOGLE_DRIVE_FOLDER_ID not set in .env file")
                    print_folder_setup_instructions()
                    return False
            else:
                print(f"❌ Service account key is missing fields: {missing_fields}")
                print_service_account_instructions()
                return False
                
        except json.JSONDecodeError:
            print("❌ Service account key file is not valid JSON")
            print_service_account_instructions()
            return False
        except Exception as e:
            print(f"❌ Error reading service account key: {e}")
            return False
    else:
        print("❌ Service account key not found")
        print_service_account_instructions()
        return False

def print_service_account_instructions():
    """Print instructions for creating a service account"""
    print("\n📋 How to create a Google Service Account:")
    print("=" * 50)
    print("1. Go to Google Cloud Console: https://console.cloud.google.com/")
    print("2. Create a new project or select existing project")
    print("3. Enable Google Drive API:")
    print("   - Go to APIs & Services > Library")
    print("   - Search for 'Google Drive API' and enable it")
    print("4. Create a service account:")
    print("   - Go to APIs & Services > Credentials")
    print("   - Click 'Create Credentials' > 'Service Account'")
    print("   - Fill in service account details")
    print("   - Click 'Create and Continue'")
    print("5. Download the service account key:")
    print("   - Go to the created service account")
    print("   - Click 'Keys' tab")
    print("   - Click 'Add Key' > 'Create new key'")
    print("   - Choose JSON format and download")
    print("6. Save the downloaded JSON file as:")
    print(f"   {Path('secrets/service_account.json').absolute()}")
    print("\n⚠️ Keep this file secure and never commit it to version control!")

def print_folder_setup_instructions():
    """Print instructions for setting up Google Drive folder"""
    print("\n📁 How to setup Google Drive folder:")
    print("=" * 50)
    print("1. Create a folder in Google Drive for the bot data")
    print("2. Right-click the folder and select 'Share'")
    print("3. Add your service account email (found in the JSON key file)")
    print("4. Give it 'Editor' permissions")
    print("5. Copy the folder ID from the URL:")
    print("   URL: https://drive.google.com/drive/folders/FOLDER_ID_HERE")
    print("6. Add this to your .env file:")
    print("   GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here")
    print("7. Set USE_GOOGLE_DRIVE=true in your .env file")

def create_sample_env():
    """Create a sample .env file with Google Drive settings"""
    env_path = Path(".env")
    
    sample_content = """# Money Printer Configuration

# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_TESTNET=false

# Google Drive Configuration
USE_GOOGLE_DRIVE=true
GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id_here

# Discord Configuration (optional)
DISCORD_WEBHOOK=your_discord_webhook_url_here
DISCORD_BOT_TOKEN=your_discord_bot_token_here
DISCORD_CHANNEL_ID=your_discord_channel_id_here

# Railway Configuration (for deployment)
RAILWAY_API_TOKEN=your_railway_api_token_here
RAILWAY_PROJECT_ID=your_railway_project_id_here
"""
    
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(sample_content)
        print(f"✅ Created sample .env file: {env_path.absolute()}")
        print("📝 Please edit this file with your actual API keys and settings")
    else:
        print("✅ .env file already exists")

def test_google_drive_connection():
    """Test Google Drive connection after setup"""
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        from src.drive_manager import EnhancedDriveManager
        
        print("\n🧪 Testing Google Drive connection...")
        
        drive_manager = EnhancedDriveManager()
        files = drive_manager.list_files_in_folder()
        
        print(f"✅ Successfully connected to Google Drive!")
        print(f"📁 Found {len(files)} files in the target folder")
        
        if files:
            print("📋 Recent files:")
            for i, file_info in enumerate(files[:5]):
                name = file_info.get('name', 'Unknown')
                size = file_info.get('size', '0')
                file_size_kb = int(size) / 1024 if size.isdigit() else 0
                print(f"   {i+1}. {name} ({file_size_kb:.1f} KB)")
        else:
            print("📝 Folder is empty (this is normal for a new setup)")
        
        return True
        
    except Exception as e:
        print(f"❌ Google Drive connection test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 Money Printer Google Drive Setup")
    print("This script will help you configure Google Drive integration")
    print()
    
    # Create sample .env if needed
    create_sample_env()
    print()
    
    # Check current setup
    if setup_google_drive():
        print("\n🧪 Testing connection...")
        if test_google_drive_connection():
            print("\n🎉 Google Drive setup is complete and working!")
            print("💡 You can now use the Money Printer with Google Drive storage")
            print("🚀 Run: python main_production.py collect --hours 1")
        else:
            print("\n⚠️ Connection test failed. Please check your setup.")
    else:
        print("\n❌ Setup incomplete. Please follow the instructions above.")
    
    print("\n" + "="*50)
    print("📖 For more help, see the README or contact support")

if __name__ == "__main__":
    main()

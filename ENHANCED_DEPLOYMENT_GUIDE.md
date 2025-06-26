# 🚀 Enhanced Google Drive & Railway Deployment Guide

## 📋 Overview

This guide covers the enhanced Google Drive integration with service account authentication, batch upload management, and Docker + Railway deployment for your crypto trading bot.

## 🔧 Enhanced Features

### ✨ **New Google Drive Manager**
- **Service Account Authentication** - No OAuth flow needed
- **Batch Upload Manager** - Intelligent batching (2-3 files per 30-60s)
- **Large File Chunking** - Handles >10MB Parquet files
- **Organized Folder Structure** - Auto-organized by category and date
- **Cancellable Operations** - Graceful handling of disconnections
- **Boot Sync** - Downloads missing files on startup

### 🚂 **Railway Integration** 
- **Usage Monitoring** - Real-time usage tracking via GraphQL API
- **Auto-shutdown** - Prevents overcharges at 450 hours
- **Discord Alerts** - Warning notifications at 400 hours

### 🐳 **Docker Deployment**
- **Production Dockerfile** - Optimized for Railway
- **Health Checks** - Built-in monitoring
- **Entrypoint Script** - Handles Drive sync on boot

---

## 🛠️ Setup Instructions

### 1. **Google Drive Service Account Setup**

#### **Step 1: Create Service Account**
```bash
# Run setup helper
python src/drive_manager.py --setup
```

Follow these steps:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the **Google Drive API**
4. Go to **Credentials** → **Create Credentials** → **Service Account**
5. Create a service account with Drive access
6. Generate and download the **JSON key file**
7. Save it as: `secrets/service_account.json`

#### **Step 2: Share Drive Folder**
1. Create a Google Drive folder for your trading data
2. Get the folder ID from the URL: `https://drive.google.com/drive/folders/[FOLDER_ID_HERE]`
3. Share the folder with the service account email (found in JSON key as `client_email`)
4. Give **Editor** permissions

#### **Step 3: Environment Configuration**
```env
# Enhanced Drive Settings
USE_GOOGLE_DRIVE=true
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
SYNC_ON_BOOT=true

# Railway Settings
RAILWAY_API_TOKEN=your_railway_token
RAILWAY_PROJECT_ID=your_project_id
USE_RAILWAY_WATCHDOG=true
```

### 2. **Railway API Setup**

#### **Get Railway API Token**
1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Go to **Account Settings** → **Tokens**
3. Create a new token
4. Add to `.env` as `RAILWAY_API_TOKEN`

#### **Get Project ID**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and get project info
railway login
railway status
```

### 3. **Test Enhanced Integration**

```bash
# Test everything
python test_enhanced_integration.py

# Test individual components
python src/drive_manager.py --test
python src/railway_watchdog.py --status
```

---

## 📁 Organized Folder Structure

The enhanced Drive manager creates an organized folder structure:

```
📁 your_drive_folder/
├── 📁 trading_data/
│   ├── 📁 models/
│   │   ├── 📁 random_forest/
│   │   ├── 📁 lightgbm/
│   │   └── 📁 neural_networks/
│   ├── 📁 trades/
│   │   ├── 📁 transactions/
│   │   ├── 📁 backtest_results/
│   │   └── 📁 performance/
│   ├── 📁 market_data/
│   │   └── 📁 scraped/
│   │       └── 📁 2025/01/26/  # Date-based
│   ├── 📁 diagnostics/
│   │   └── 📁 2025/01/26/      # Date-based
│   └── 📁 stats/
│       └── 📁 daily/2025/01/   # Date-based
├── 📁 logs/
│   └── 📁 2025/01/26/          # Date-based
└── 📁 backups/
    ├── 📁 configurations/
    └── 📁 critical_files/
```

---

## 🚀 Docker Deployment

### **Dockerfile Features**
- **Python 3.11** optimized base
- **Multi-stage builds** for efficiency
- **Health checks** built-in
- **Non-root user** for security
- **Railway optimization**

### **Build and Test Locally**
```bash
# Build Docker image
docker build -t crypto-trading-bot .

# Test locally
docker run -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/secrets:/app/secrets \
  crypto-trading-bot
```

### **Railway Deployment**

#### **Method 1: Auto-Deploy from GitHub**
1. Connect your GitHub repo to Railway
2. Railway will auto-detect the Dockerfile
3. Set environment variables in Railway dashboard
4. Upload `service_account.json` as a volume

#### **Method 2: Railway CLI**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link [your-project-id]

# Set environment variables
railway variables set USE_GOOGLE_DRIVE=true
railway variables set GOOGLE_DRIVE_FOLDER_ID=your_folder_id
railway variables set SYNC_ON_BOOT=true

# Upload service account as volume
# (Do this through Railway dashboard under "Variables" → "Volume")

# Deploy
railway up
```

---

## 🎮 Enhanced Discord Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/usage_status` | Railway usage & billing | Real-time usage tracking |
| `/drive_status` | Enhanced Drive status | Service account, batch stats |
| `/drive_sync` | Manual sync trigger | Organized upload batching |

### **Enhanced Status Display**
- **Green** 🟢: Normal usage (< 400 hours)
- **Orange** 🟡: Warning zone (400-450 hours)
- **Red** 🔴: Critical - auto-shutdown (450+ hours)

---

## ⚙️ Advanced Configuration

### **Batch Upload Settings**
```python
# In src/drive_manager.py
CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
MAX_BATCH_SIZE = 3             # Files per batch
MIN_BATCH_INTERVAL = 30        # Minimum seconds between batches
MAX_BATCH_INTERVAL = 60        # Maximum seconds between batches
```

### **File Categories**
- **Models**: `category="models"`, `subcategory="random_forest|lightgbm|neural_networks"`
- **Trades**: `category="trades"`, `subcategory="transactions|backtest_results|performance"`
- **Market Data**: `category="market_data"`, `subcategory="scraped"`, `date_based=True`
- **Diagnostics**: `category="diagnostics"`, `subcategory="logs|errors|system_stats"`, `date_based=True`
- **Stats**: `category="stats"`, `subcategory="daily|weekly|monthly"`, `date_based=True`

### **Priority System**
- **Priority 5**: Critical files (immediate upload)
- **Priority 3**: Model files
- **Priority 2**: Trade data, stats
- **Priority 1**: Diagnostics, logs

---

## 🛡️ Production Considerations

### **Security**
- ✅ Service account authentication (no OAuth tokens)
- ✅ Non-root Docker user
- ✅ Environment variable management
- ✅ Secure volume mounting for keys

### **Reliability**
- ✅ Automatic retry on failures (3 attempts)
- ✅ Graceful cancellation of operations
- ✅ Health checks and monitoring
- ✅ Error handling and logging

### **Performance**
- ✅ Batch uploads for efficiency
- ✅ File deduplication with SHA256
- ✅ Chunked uploads for large files
- ✅ Rate limiting to avoid API limits

### **Monitoring**
- ✅ Real-time usage tracking
- ✅ Discord notifications
- ✅ Comprehensive logging
- ✅ Health check endpoints

---

## 🧪 Testing & Validation

```bash
# Full integration test
python test_enhanced_integration.py

# Individual component tests
python src/drive_manager.py --test
python src/railway_watchdog.py --check-once
python docker/health_check.py

# Background services test
python background_services.py
```

### **Expected Test Results**
```
✅ Enhanced Drive Manager Test PASSED
✅ Background Services Test PASSED  
✅ Railway Integration Test PASSED
✅ Docker Health Check Test PASSED

🎉 All tests passed! Enhanced integration is ready.
```

---

## 🚨 Troubleshooting

### **Common Issues**

**Service Account Authentication Failed**
```bash
# Check service account file
ls -la secrets/service_account.json

# Verify folder sharing
python src/drive_manager.py --test
```

**Railway API Issues**
```bash
# Verify token
python src/railway_watchdog.py --status

# Check project ID
railway status
```

**Docker Build Issues**
```bash
# Check Dockerfile syntax
docker build --no-cache -t crypto-trading-bot .

# Verify health check
docker run --rm crypto-trading-bot python docker/health_check.py
```

**Batch Upload Not Working**
```bash
# Check batch manager status
python -c "
from src.drive_manager import get_drive_manager
manager = get_drive_manager()
if manager.batch_manager:
    print(manager.batch_manager.get_stats())
"
```

---

## 📊 Monitoring & Maintenance

### **Daily Checks**
- Monitor Railway usage via Discord `/usage_status`
- Check Drive sync status via `/drive_status`
- Review logs for any errors

### **Weekly Maintenance**
- Review uploaded file organization
- Check batch upload efficiency
- Validate health check responses

### **Monthly Review**
- Analyze Railway usage patterns
- Review Drive storage usage
- Update service account permissions if needed

---

## 🎉 Ready for Production!

Your enhanced crypto trading bot now features:

✅ **Production-Ready Drive Integration** with service accounts  
✅ **Intelligent Batch Upload Management** with rate limiting  
✅ **Organized File Structure** with automatic categorization  
✅ **Railway Usage Protection** with auto-shutdown  
✅ **Docker Deployment** optimized for Railway  
✅ **Comprehensive Monitoring** via Discord commands  

The bot is now ready for 24/7 production deployment with enterprise-grade reliability and monitoring! 🚀

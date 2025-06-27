🔑 **GOOGLE DRIVE SERVICE ACCOUNT SETUP GUIDE**

To enable Google Drive upload, you need a service account key file.

## 📋 **Step 1: Create Google Cloud Project**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable **Google Drive API**

## 🔐 **Step 2: Create Service Account**
1. Go to **IAM & Admin → Service Accounts**
2. Click **Create Service Account**
3. Name it: `money-printer-drive`
4. Click **Create and Continue**
5. Skip role assignment for now
6. Click **Done**

## 🗝️ **Step 3: Generate Key**
1. Click on your new service account
2. Go to **Keys** tab
3. Click **Add Key → Create New Key**
4. Choose **JSON** format
5. Download the file

## 📁 **Step 4: Install Key**
1. **Rename** the downloaded file to: `service_account.json`
2. **Place** it in: `Z:\money_printer\secrets\service_account.json`
3. **NEVER** commit this file to git (it's in .gitignore)

## 🔗 **Step 5: Share Drive Folder**
1. Open your Google Drive folder: `1tIujkkmknMOTKprDGhZiab3FYF_Qzpmj`
2. Click **Share**
3. Add the service account email (from the JSON file)
4. Give it **Editor** permissions

## ✅ **Step 6: Test**
Run: `python test_drive_and_balance.py`
Should show: ✅ Google Drive Access: PASSED

---

**🔄 After setup:** Data scraper will automatically upload files to Google Drive!

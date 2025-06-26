@echo off
REM Discord Bot Launcher for Windows
echo ============================================
echo         DISCORD BOT LAUNCHER
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ❌ .env file not found! Please create it first.
    echo Copy .env.example to .env and fill in your values.
    pause
    exit /b 1
)

echo Choose which Discord bot to start:
echo.
echo 1. Trading Bot (Control trading operations)
echo 2. Data Collector Bot (Control data scraping)  
echo 3. Both Bots
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🤖 Starting Trading Discord Bot...
    cd src\trading_bot
    python discord_trader_bot.py
    cd ..\..
) else if "%choice%"=="2" (
    echo.
    echo 📊 Starting Data Collector Discord Bot...
    cd src\data_collector
    python discord_bot.py
    cd ..\..
) else if "%choice%"=="3" (
    echo.
    echo 🚀 Starting both Discord bots...
    python start_discord_bots.py
) else if "%choice%"=="4" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    pause
    goto :eof
)

pause

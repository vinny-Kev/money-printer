#!/bin/bash
# Optimized entrypoint script for Railway deployment
# Minimal setup for unified Discord bot

set -e

echo "🤖 Starting Unified Discord Bot on Railway..."
echo "📅 $(date)"

# Environment variables
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create basic directory structure
log "📁 Creating directory structure..."
mkdir -p /app/data/{models,transactions,diagnostics,scraped_data}
mkdir -p /app/logs

# Quick health check
log "🔍 Running health check..."
python -c "
import sys
print('✅ Python environment ready')
try:
    import discord
    print('✅ Discord.py available')
except ImportError as e:
    print(f'❌ Discord.py missing: {e}')
    sys.exit(1)
"

# Check Discord token
log "🔑 Validating Discord configuration..."
if [ -z "${DISCORD_BOT_TOKEN}" ]; then
    log "❌ DISCORD_BOT_TOKEN not set!"
    exit 1
fi

if [ -z "${DISCORD_USER_ID}" ]; then
    log "⚠️ DISCORD_USER_ID not set - bot will accept commands from all users"
else
    log "✅ Discord bot configured for user: ${DISCORD_USER_ID}"
fi

# Start the unified Discord bot
log "🤖 Starting Unified Discord Bot..."
cd /app

# Execute the bot
exec python src/unified_discord_bot.py
print('✅ Python environment ready')
try:
    import discord
    print('✅ Discord.py available')
except ImportError as e:
    print(f'❌ Discord.py missing: {e}')
    sys.exit(1)
"
        print('📥 Downloading missing files...')
        results = manager.download_missing_files()
        print(f'✅ Download complete: {results}')
    else:
        print('❌ Drive not authenticated')
except Exception as e:
    print(f'❌ Drive sync failed: {e}')
" || log "⚠️ Drive sync failed, continuing..."
    else
        log "⏭️ Drive sync disabled"
    fi
}

# Function to setup directories
setup_directories() {
    log "📁 Setting up directories..."
    
    mkdir -p /app/data/models
    mkdir -p /app/data/transactions
    mkdir -p /app/data/diagnostics
    mkdir -p /app/data/scraped_data/parquet_files
    mkdir -p /app/logs
    mkdir -p /app/secrets
    
    log "✅ Directories ready"
}

# Function to check Railway environment
check_railway_env() {
    log "🚂 Checking Railway environment..."
    
    if [ -n "$RAILWAY_ENVIRONMENT" ]; then
        log "✅ Running on Railway environment: $RAILWAY_ENVIRONMENT"
        
        # Set Railway-specific configurations
        export DEPLOY_ENV="railway"
        export ENVIRONMENT="production"
        
        # Railway provides these automatically
        if [ -n "$PORT" ]; then
            log "✅ Railway PORT detected: $PORT"
        fi
        
        if [ -n "$RAILWAY_PROJECT_ID" ]; then
            log "✅ Railway project: $RAILWAY_PROJECT_ID"
        fi
    else
        log "⚠️ Not running on Railway, using local configuration"
        export DEPLOY_ENV="local"
        export ENVIRONMENT="development"
    fi
}

# Function to validate environment
validate_environment() {
    log "🔍 Validating environment..."
    
    # Check required environment variables
    required_vars=("DISCORD_BOT_TOKEN" "BINANCE_API_KEY" "BINANCE_SECRET_KEY")
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log "❌ Required environment variable $var is not set"
            exit 1
        else
            log "✅ $var is configured"
        fi
    done
    
    # Check optional but important variables
    optional_vars=("RAILWAY_API_TOKEN" "GOOGLE_DRIVE_FOLDER_ID" "DISCORD_WEBHOOK_URL")
    
    for var in "${optional_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log "⚠️ Optional variable $var is not set"
        else
            log "✅ $var is configured"
        fi
    done
}

# Function to start health monitoring
start_health_monitoring() {
    log "💓 Starting health monitoring..."
    
    # Start background health check
    python -c "
import sys
sys.path.append('/app')
import asyncio
from background_services import start_background_services

try:
    print('🔄 Starting background services...')
    asyncio.run(start_background_services())
except Exception as e:
    print(f'❌ Background services failed: {e}')
" &
    
    log "✅ Background services started"
}

# Function to start Discord bot
start_discord_bot() {
    log "🤖 Starting unified Discord bot for trading and scraper control..."
    
    # Check if Discord token is available
    if [ -z "$DISCORD_BOT_TOKEN" ]; then
        log "⚠️ DISCORD_BOT_TOKEN not set, skipping Discord bot"
        return 0
    fi
    
    # Start the unified Discord bot
    cd /app/src
    python unified_discord_bot.py &
    DISCORD_PID=$!
    
    log "✅ Discord bot started (PID: $DISCORD_PID)"
    
    # Store PID for cleanup
    echo $DISCORD_PID > /app/discord_bot.pid
    
    cd /app
}

# Function to start main application
start_main_application() {
    log "🚀 Starting main trading application..."
    
    # Determine which mode to run
    case "${APP_MODE:-discord}" in
        "discord")
            log "🎮 Running in Discord control mode"
            start_discord_bot
            
            # Keep container alive and monitor Discord bot
            while true; do
                if [ -f "/app/discord_bot.pid" ]; then
                    DISCORD_PID=$(cat /app/discord_bot.pid)
                    if ! kill -0 $DISCORD_PID 2>/dev/null; then
                        log "⚠️ Discord bot died, restarting..."
                        start_discord_bot
                    fi
                fi
                sleep 30
            done
            ;;
        "trading")
            log "💰 Running in direct trading mode"
            python main.py trade
            ;;
        "collect")
            log "📊 Running in data collection mode"
            python main.py collect
            ;;
        "train")
            log "🤖 Running in model training mode"
            python main.py train --model random_forest
            ;;
        *)
            log "❌ Unknown APP_MODE: ${APP_MODE}"
            log "Valid modes: discord, trading, collect, train"
            exit 1
            ;;
    esac
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up processes..."
    
    if [ -f "/app/discord_bot.pid" ]; then
        DISCORD_PID=$(cat /app/discord_bot.pid)
        if kill -0 $DISCORD_PID 2>/dev/null; then
            log "🛑 Stopping Discord bot..."
            kill $DISCORD_PID
        fi
        rm -f /app/discord_bot.pid
    fi
    
    log "👋 Cleanup complete"
}

# Set trap for cleanup
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log "🚀 Starting entrypoint script..."
    
    # Setup
    check_railway_env
    setup_directories
    validate_environment
    
    # Google Drive sync on boot
    if [ "${SYNC_ON_BOOT:-true}" = "true" ]; then
        sync_drive_on_boot
    else
        log "⏭️ Skipping boot sync (SYNC_ON_BOOT=false)"
    fi
    
    # Start health monitoring
    start_health_monitoring
    
    # Wait a moment for services to initialize
    sleep 5
    
    # Start main application
    start_main_application
}

# Run main function
main "$@"

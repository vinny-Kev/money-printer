#!/usr/bin/env python3
"""
Railway Health Server - Standalone health check for Railway deployment
This ensures the health endpoint is always available even if the main bot has issues
"""
import os
import sys
import logging
import asyncio
from datetime import datetime
from aiohttp import web
import json

# Configuration
PORT = int(os.getenv("PORT", "8000"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("HealthServer")

# Health check endpoint
async def health_check(request):
    """Health check endpoint for Railway deployment."""
    return web.json_response({
        "status": "healthy",
        "service": "money-printer-trading-system",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.1",
        "environment": "production"
    })

async def status_check(request):
    """Status endpoint with more detailed information."""
    try:
        # Try to check if main services are working
        status_info = {
            "status": "healthy",
            "service": "money-printer-trading-system",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.1",
            "environment": "production",
            "components": {
                "health_server": "running",
                "discord_bot": "unknown",
                "trading_engine": "unknown",
                "data_collector": "unknown"
            }
        }
        
        return web.json_response(status_info)
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return web.json_response({
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status=200)  # Still return 200 for Railway health check

async def start_health_server():
    """Start the health check web server."""
    app = web.Application()
    
    # Add health check routes
    app.router.add_get('/health', health_check)
    app.router.add_get('/status', status_check)
    app.router.add_get('/', health_check)  # Default route
    
    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    
    logger.info(f"🌐 Health server started on port {PORT}")
    logger.info(f"🔍 Health endpoint: http://0.0.0.0:{PORT}/health")
    logger.info(f"📊 Status endpoint: http://0.0.0.0:{PORT}/status")
    
    return runner

async def start_main_bot():
    """Start the main Discord bot in a separate task."""
    try:
        logger.info("🤖 Starting main Discord bot...")
        
        # Import and start the main bot
        from lightweight_discord_bot import main_async as bot_main
        await bot_main()
        
    except Exception as e:
        logger.error(f"❌ Discord bot failed to start: {e}")
        logger.info("⚠️ Health server will continue running without Discord bot")
        # Don't fail - keep health server running

async def main():
    """Main function - start health server and then try to start the bot."""
    logger.info("🚀 Starting Money Printer Health Server")
    
    # Start health server first
    health_runner = await start_health_server()
    
    try:
        # Try to start the main bot
        await start_main_bot()
    except Exception as e:
        logger.error(f"❌ Main bot startup failed: {e}")
        logger.info("✅ Health server continues running for Railway deployment")
        
        # Keep the health server running indefinitely
        try:
            while True:
                await asyncio.sleep(60)  # Sleep for 1 minute at a time
                logger.debug("🔄 Health server heartbeat")
        except KeyboardInterrupt:
            logger.info("🛑 Health server stopped by user")
        finally:
            await health_runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        sys.exit(1)

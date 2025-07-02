#!/usr/bin/env python3
"""
Railway Production Server - Production-ready health and Discord bot server
Handles environment variables gracefully and ensures health endpoint is always available
"""
import os
import sys
import logging
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

# Configuration
PORT = int(os.getenv("PORT", "8000"))
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ProductionServer")

# Global status tracking
service_status = {
    "health_server": "starting",
    "discord_bot": "not_started",
    "environment_check": "pending"
}

class ProductionHealthHandler(BaseHTTPRequestHandler):
    """Production HTTP handler for health checks with detailed status."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path in ['/health', '/', '/status']:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "status": "healthy",
                "service": "money-printer-trading-system",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.1",
                "environment": "production",
                "port": PORT,
                "path": self.path,
                "uptime_seconds": int(time.time() - start_time),
                "services": service_status.copy(),
                "environment_vars": {
                    "PORT": PORT,
                    "DISCORD_TOKEN_SET": bool(DISCORD_TOKEN),
                    "DISCORD_USER_ID_SET": bool(DISCORD_USER_ID),
                    "PYTHONPATH": os.getenv("PYTHONPATH", "not_set")
                }
            }
            
            self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Not Found", "path": self.path}
            self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"HTTP {format % args}")

def check_environment():
    """Check if environment is properly configured."""
    logger.info("🔍 Checking production environment...")
    
    service_status["environment_check"] = "running"
    
    if not DISCORD_TOKEN:
        logger.warning("⚠️ DISCORD_BOT_TOKEN not set - Discord bot will not start")
        service_status["discord_bot"] = "disabled_no_token"
    else:
        logger.info("✅ DISCORD_BOT_TOKEN is set")
        service_status["discord_bot"] = "token_available"
    
    if not DISCORD_USER_ID:
        logger.warning("⚠️ DISCORD_USER_ID not set - bot will have limited functionality")
    else:
        logger.info("✅ DISCORD_USER_ID is set")
    
    logger.info(f"✅ PORT set to: {PORT}")
    logger.info(f"✅ PYTHONPATH: {os.getenv('PYTHONPATH', 'default')}")
    
    service_status["environment_check"] = "completed"

def start_discord_bot():
    """Start Discord bot if environment allows."""
    def run_bot():
        try:
            if not DISCORD_TOKEN:
                logger.info("ℹ️ Discord bot disabled - no token provided")
                service_status["discord_bot"] = "disabled"
                return
                
            logger.info("🤖 Starting Discord bot...")
            service_status["discord_bot"] = "starting"
            
            # Add paths for imports
            sys.path.append('/app')
            sys.path.append('/app/src')
            
            # Import and start Discord bot
            from src.lightweight_discord_bot import main
            service_status["discord_bot"] = "running"
            main()
            
        except ImportError as e:
            logger.error(f"❌ Discord bot import failed: {e}")
            service_status["discord_bot"] = "import_failed"
        except Exception as e:
            logger.error(f"❌ Discord bot failed: {e}")
            service_status["discord_bot"] = "error"
        
        logger.info("ℹ️ Discord bot thread ended")
    
    # Start Discord bot in background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    return bot_thread

def start_health_server():
    """Start the health server."""
    logger.info(f"🚀 Starting production health server on port {PORT}")
    service_status["health_server"] = "running"
    
    server = HTTPServer(('0.0.0.0', PORT), ProductionHealthHandler)
    logger.info(f"🌐 Health server running on http://0.0.0.0:{PORT}")
    logger.info(f"🔍 Health endpoint: http://0.0.0.0:{PORT}/health")
    logger.info(f"📊 Status endpoint: http://0.0.0.0:{PORT}/status")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Health server stopped by user")
    finally:
        service_status["health_server"] = "stopped"
        server.shutdown()

# Track start time
start_time = time.time()

def main():
    """Main function."""
    logger.info("🚀 Starting Money Printer Production Server")
    logger.info("=" * 50)
    
    # Check environment
    check_environment()
    
    # Start Discord bot in background
    discord_thread = start_discord_bot()
    
    # Start health server (blocking)
    start_health_server()

if __name__ == "__main__":
    main()

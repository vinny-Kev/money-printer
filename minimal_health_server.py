#!/usr/bin/env python3
"""
Minimal Railway Health Server - Ultra-simple health check for Railway deployment
This is a standalone server that only provides health endpoints with minimal dependencies
"""
import os
import sys
import logging
import asyncio
import json
from datetime import datetime

# Use built-in http.server for minimal dependencies
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

# Configuration
PORT = int(os.getenv("PORT", "8000"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MinimalHealthServer")

class HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path in ['/health', '/', '/status']:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "status": "healthy",
                "service": "money-printer-trading-system",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.1",
                "environment": "production",
                "port": PORT,
                "path": self.path
            }
            
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"HTTP {format % args}")

def start_health_server():
    """Start the minimal health server."""
    logger.info(f"🚀 Starting minimal health server on port {PORT}")
    
    server = HTTPServer(('0.0.0.0', PORT), HealthHandler)
    logger.info(f"🌐 Health server running on http://0.0.0.0:{PORT}")
    logger.info(f"🔍 Health endpoint: http://0.0.0.0:{PORT}/health")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Health server stopped by user")
        server.shutdown()

def start_background_services():
    """Start background services (Discord bot, trading, etc.) in a separate thread."""
    def run_services():
        try:
            logger.info("🤖 Attempting to start background services...")
            
            # Try to start the Discord bot
            sys.path.append('/app/src')
            from lightweight_discord_bot import main
            main()
            
        except Exception as e:
            logger.error(f"❌ Background services failed: {e}")
            logger.info("✅ Health server continues running for Railway deployment")
    
    # Start background services in a separate thread
    services_thread = threading.Thread(target=run_services, daemon=True)
    services_thread.start()
    
    return services_thread

def main():
    """Main function."""
    logger.info("🚀 Starting Money Printer Minimal Health Server")
    
    # Start background services (non-blocking)
    services_thread = start_background_services()
    
    # Start health server (blocking)
    start_health_server()

if __name__ == "__main__":
    main()

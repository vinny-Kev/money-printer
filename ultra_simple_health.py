#!/usr/bin/env python3
"""
Ultra-Simple Health Server for Railway - Zero Dependencies
This is the most basic possible health server that will always work
"""
import os
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

# Configuration
PORT = int(os.getenv("PORT", "8000"))

class UltraSimpleHealthHandler(BaseHTTPRequestHandler):
    """Ultra-simple health check handler with zero external dependencies."""
    
    def do_GET(self):
        """Handle all GET requests with a simple health response."""
        # Always return 200 OK for any path
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Simple health response
        response = {
            "status": "healthy",
            "service": "money-printer",
            "timestamp": datetime.now().isoformat(),
            "port": PORT,
            "path": self.path,
            "message": "Ultra-simple health server running"
        }
        
        try:
            response_json = json.dumps(response)
        except:
            # Fallback if JSON fails
            response_json = '{"status": "healthy", "service": "money-printer"}'
            
        self.wfile.write(response_json.encode('utf-8'))
    
    def do_HEAD(self):
        """Handle HEAD requests (some health checkers use this)."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Simple logging."""
        print(f"[{datetime.now()}] {format % args}")

def main():
    """Start the ultra-simple health server."""
    print(f"🚀 Starting ultra-simple health server on port {PORT}")
    print(f"📍 Health endpoint: http://0.0.0.0:{PORT}/health")
    print(f"📍 Any path will return healthy status")
    
    try:
        server = HTTPServer(('0.0.0.0', PORT), UltraSimpleHealthHandler)
        print(f"✅ Server ready on port {PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

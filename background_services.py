#!/usr/bin/env python3
"""
Background Services Manager
Runs Railway watchdog and Drive sync as background services.
"""

import os
import sys
import asyncio
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import LOGS_DIR, USE_GOOGLE_DRIVE
from src.railway_watchdog import get_railway_watchdog
from src.drive_manager import get_drive_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "background_services.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except AttributeError:
        # Some file objects don't have .buffer attribute
        pass

class BackgroundServices:
    """Manages background services for Railway and Drive monitoring"""
    
    def __init__(self):
        self.running = True
        self.railway_watchdog = None
        self.drive_manager = None
        self.tasks = []
          # Initialize services
        try:
            self.railway_watchdog = get_railway_watchdog()
            logger.info("✅ Railway watchdog initialized")
        except Exception as e:
            logger.warning(f"Railway watchdog not available: {e}")
            
        if USE_GOOGLE_DRIVE:
            try:
                self.drive_manager = get_drive_manager()
                logger.info("✅ Drive manager initialized")
            except Exception as e:
                logger.warning(f"Drive manager not available: {e}")
    
    async def start_railway_monitoring(self):
        """Start Railway usage monitoring"""
        if not self.railway_watchdog:
            logger.warning("Railway watchdog not available")
            return
            
        try:
            logger.info("🚂 Starting Railway usage monitoring...")
            await self.railway_watchdog.start_monitoring()
        except Exception as e:            logger.error(f"Railway monitoring failed: {e}")
    
    async def start_drive_sync(self):
        """Start periodic Drive sync"""
        if not self.drive_manager or not USE_GOOGLE_DRIVE:
            logger.info("Drive sync disabled or not available")
            return
            
        logger.info("📁 Starting periodic Drive sync...")
        
        # Sync every 30 minutes
        sync_interval = 30 * 60  # 30 minutes in seconds
        
        while self.running:
            try:
                # Sync trading data
                results = self.drive_manager.sync_trading_data()
                
                if "error" not in results:
                    total_synced = sum([
                        results.get('models', 0),
                        results.get('trades', 0),
                        results.get('market_data', 0),
                        results.get('diagnostics', 0),
                        results.get('stats', 0),
                        results.get('logs', 0)
                    ])
                    
                    if total_synced > 0:
                        logger.info(f"📁 Queued {total_synced} files for Drive sync")
                    else:
                        logger.debug("📁 Drive sync: no new files to sync")
                else:
                    logger.error(f"📁 Drive sync failed: {results['error']}")
                
                # Wait for next sync
                await asyncio.sleep(sync_interval)
                
            except Exception as e:
                logger.error(f"Drive sync error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def health_check(self):
        """Periodic health check of all services"""
        logger.info("❤️ Starting health check service...")
        
        check_interval = 10 * 60  # 10 minutes
        
        while self.running:
            try:
                # Check Railway service
                if self.railway_watchdog:
                    usage = await self.railway_watchdog.check_usage_once()
                    if usage:
                        logger.debug(f"Railway health: {usage.usage_percentage:.1f}% used")
                  # Check Drive service
                if self.drive_manager and USE_GOOGLE_DRIVE:
                    status = self.drive_manager.get_status()
                    if status['authenticated']:
                        logger.debug("Drive health: OK")
                    else:
                        logger.warning("Drive health: Authentication failed")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("🛑 Shutdown signal received")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
    
    async def start_all_services(self):
        """Start all background services"""
        logger.info("🚀 Starting background services...")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        # Create tasks for each service
        tasks = []
        
        # Railway monitoring
        if self.railway_watchdog:
            task = asyncio.create_task(self.start_railway_monitoring())
            tasks.append(task)
          # Drive sync
        if self.drive_manager and USE_GOOGLE_DRIVE:
            task = asyncio.create_task(self.start_drive_sync())
            tasks.append(task)
        
        # Health check
        task = asyncio.create_task(self.health_check())
        tasks.append(task)
        
        self.tasks = tasks
        
        if not tasks:
            logger.warning("No services to start")
            return
        
        logger.info(f"✅ Started {len(tasks)} background services")
        
        try:
            # Wait for all tasks to complete (or be cancelled)
            await asyncio.gather(*tasks, return_exceptions=True)
        except KeyboardInterrupt:
            logger.info("Services stopped by user")
        finally:
            logger.info("🏁 All background services stopped")

async def main():
    """Main entry point"""
    services = BackgroundServices()
    await services.start_all_services()

# Export function for use in entrypoint script
async def start_background_services():
    """Start background services - wrapper for external import"""
    return await main()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Background services stopped by user")
    except Exception as e:
        logger.error(f"Background services error: {e}")
        sys.exit(1)

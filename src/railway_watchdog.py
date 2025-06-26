#!/usr/bin/env python3
"""
Railway Usage Watchdog
Monitors Railway app usage and prevents billing overages by automatically shutting down.
"""

import os
import sys
import time
import json
import logging
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import (
    RAILWAY_API_TOKEN, RAILWAY_PROJECT_ID, RAILWAY_MAX_USAGE_HOURS,
    RAILWAY_WARNING_HOURS, RAILWAY_CHECK_INTERVAL, LOGS_DIR
)
from src.discord_notifications import send_general_notification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "railway_watchdog.log", encoding='utf-8'),
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

@dataclass
class RailwayUsage:
    """Railway usage statistics"""
    current_hours: float
    limit_hours: float
    remaining_hours: float
    usage_percentage: float
    billing_cycle_start: str
    billing_cycle_end: str
    estimated_monthly_cost: float = 0.0

class RailwayWatchdog:
    """Monitor Railway usage and prevent billing overages"""
    
    def __init__(self, api_token: str = None, project_id: str = None):
        self.api_token = api_token or RAILWAY_API_TOKEN
        self.project_id = project_id or RAILWAY_PROJECT_ID
        self.max_hours = RAILWAY_MAX_USAGE_HOURS
        self.warning_hours = RAILWAY_WARNING_HOURS
        self.check_interval = RAILWAY_CHECK_INTERVAL * 60  # Convert to seconds
        
        if not self.api_token or not self.project_id:
            raise ValueError("Railway API token and project ID are required")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        self.last_warning_sent = None
        self.shutdown_initiated = False
        
    def get_railway_usage(self) -> Optional[RailwayUsage]:
        """
        Fetch current Railway usage via GraphQL API
        Returns RailwayUsage object or None if failed
        """
        try:
            # Railway GraphQL endpoint
            url = "https://backboard.railway.app/graphql/v2"
            
            # GraphQL query to get project usage
            query = """
            query GetProjectUsage($projectId: String!) {
                project(id: $projectId) {
                    id
                    name
                    usage {
                        current
                        estimated
                        hardLimit
                        softLimit
                    }
                    subscriptionPlanLimit
                    subscriptionType
                }
            }
            """
            
            variables = {"projectId": self.project_id}
            
            response = requests.post(
                url,
                headers=self.headers,
                json={"query": query, "variables": variables},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Railway API request failed: {response.status_code} - {response.text}")
                return None
                
            data = response.json()
            
            if "errors" in data:
                logger.error(f"Railway API returned errors: {data['errors']}")
                return None
                
            project_data = data.get("data", {}).get("project", {})
            usage_data = project_data.get("usage", {})
            
            if not usage_data:
                logger.warning("No usage data found in Railway response")
                return None
                
            # Convert usage from minutes to hours (Railway returns minutes)
            current_minutes = usage_data.get("current", 0)
            current_hours = current_minutes / 60.0
            
            # Calculate billing cycle (Railway billing cycles are monthly)
            now = datetime.utcnow()
            billing_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                billing_end = billing_start.replace(year=now.year + 1, month=1)
            else:
                billing_end = billing_start.replace(month=now.month + 1)
            
            remaining_hours = max(0, self.max_hours - current_hours)
            usage_percentage = (current_hours / self.max_hours) * 100
            
            usage = RailwayUsage(
                current_hours=current_hours,
                limit_hours=self.max_hours,
                remaining_hours=remaining_hours,
                usage_percentage=usage_percentage,
                billing_cycle_start=billing_start.isoformat(),
                billing_cycle_end=billing_end.isoformat(),
                estimated_monthly_cost=usage_data.get("estimated", 0) / 100  # Convert cents to dollars
            )
            
            logger.info(f"Railway usage: {current_hours:.2f}/{self.max_hours} hours ({usage_percentage:.1f}%)")
            return usage
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching Railway usage: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Railway usage: {e}")
            return None
    
    async def send_usage_warning(self, usage: RailwayUsage):
        """Send Discord warning about high usage"""
        try:
            message = f"""⚠️ **Railway Usage Warning**

📊 **Current Usage**: {usage.current_hours:.2f} / {usage.limit_hours} hours ({usage.usage_percentage:.1f}%)
⏰ **Remaining**: {usage.remaining_hours:.2f} hours
💰 **Estimated Cost**: ${usage.estimated_monthly_cost:.2f}

🚨 **Action Required**: Usage is approaching the limit. Consider optimizing or shutting down non-essential services."""

            await send_general_notification(message)
            self.last_warning_sent = datetime.utcnow()
            logger.warning(f"Usage warning sent: {usage.usage_percentage:.1f}% used")
            
        except Exception as e:
            logger.error(f"Failed to send usage warning: {e}")
    
    async def emergency_shutdown(self, usage: RailwayUsage):
        """Initiate emergency shutdown to prevent billing overage"""
        try:
            self.shutdown_initiated = True
            
            message = f"""🚨 **EMERGENCY SHUTDOWN INITIATED**

📊 **Final Usage**: {usage.current_hours:.2f} / {usage.limit_hours} hours ({usage.usage_percentage:.1f}%)
💰 **Estimated Cost**: ${usage.estimated_monthly_cost:.2f}

🛑 **Reason**: Usage limit reached. Shutting down to prevent billing overage.
🔄 **Recovery**: Application will need manual restart."""

            await send_general_notification(message)
            logger.critical(f"Emergency shutdown initiated: {usage.usage_percentage:.1f}% usage reached")
            
            # Give time for notification to send
            await asyncio.sleep(5)
            
            # Initiate shutdown
            logger.critical("Shutting down application to prevent billing overage...")
            os._exit(1)  # Force exit
            
        except Exception as e:
            logger.error(f"Failed during emergency shutdown: {e}")
            # Still try to exit
            os._exit(1)
    
    async def check_usage_once(self) -> Optional[RailwayUsage]:
        """Check usage once and take appropriate action"""
        usage = self.get_railway_usage()
        
        if usage is None:
            logger.warning("Failed to get Railway usage data")
            return None
            
        # Check for emergency shutdown
        if usage.current_hours >= self.max_hours:
            await self.emergency_shutdown(usage)
            return usage
            
        # Check for warning threshold
        if (usage.current_hours >= self.warning_hours and 
            (self.last_warning_sent is None or 
             datetime.utcnow() - self.last_warning_sent > timedelta(hours=1))):
            await self.send_usage_warning(usage)
            
        return usage
    
    async def start_monitoring(self):
        """Start continuous usage monitoring"""
        logger.info(f"Starting Railway usage monitoring (checking every {self.check_interval//60} minutes)")
        logger.info(f"Limits: Warning at {self.warning_hours}h, Shutdown at {self.max_hours}h")
        
        while not self.shutdown_initiated:
            try:
                await self.check_usage_once()
                await asyncio.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Railway watchdog stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in usage monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
                
    def get_usage_status(self) -> Dict:
        """Get current usage status for Discord commands"""
        usage = self.get_railway_usage()
        
        if usage is None:
            return {
                "error": "Failed to fetch Railway usage data",
                "status": "unknown"
            }
            
        status = "normal"
        if usage.current_hours >= self.max_hours:
            status = "critical"
        elif usage.current_hours >= self.warning_hours:
            status = "warning"
            
        return {
            "status": status,
            "current_hours": usage.current_hours,
            "limit_hours": usage.limit_hours,
            "remaining_hours": usage.remaining_hours,
            "usage_percentage": usage.usage_percentage,
            "estimated_cost": usage.estimated_monthly_cost,
            "billing_cycle_start": usage.billing_cycle_start,
            "billing_cycle_end": usage.billing_cycle_end
        }

# Global instance
_railway_watchdog = None

def get_railway_watchdog() -> RailwayWatchdog:
    """Get or create Railway watchdog singleton"""
    global _railway_watchdog
    if _railway_watchdog is None:
        _railway_watchdog = RailwayWatchdog()
    return _railway_watchdog

async def main():
    """CLI entry point for Railway watchdog"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Railway Usage Watchdog")
    parser.add_argument("--check-once", action="store_true", help="Check usage once and exit")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--status", action="store_true", help="Show current usage status")
    
    args = parser.parse_args()
    
    try:
        watchdog = get_railway_watchdog()
        
        if args.check_once:
            usage = await watchdog.check_usage_once()
            if usage:
                print(f"Usage: {usage.current_hours:.2f}/{usage.limit_hours}h ({usage.usage_percentage:.1f}%)")
            else:
                print("Failed to get usage data")
                
        elif args.status:
            status = watchdog.get_usage_status()
            if "error" in status:
                print(f"Error: {status['error']}")
            else:
                print(f"Status: {status['status']}")
                print(f"Usage: {status['current_hours']:.2f}/{status['limit_hours']}h ({status['usage_percentage']:.1f}%)")
                print(f"Remaining: {status['remaining_hours']:.2f}h")
                print(f"Estimated Cost: ${status['estimated_cost']:.2f}")
                
        elif args.monitor:
            await watchdog.start_monitoring()
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Railway watchdog stopped by user")
    except Exception as e:
        logger.error(f"Railway watchdog error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

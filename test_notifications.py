"""
Test script for Discord notification system
"""
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from discord_notifications import (
    send_scraper_notification,
    send_trainer_notification,
    send_trader_notification,
    send_general_notification
)

def test_notifications():
    """Test all notification types"""
    print("Testing Discord notification system...")
    
    # Test general notification
    print("\n1. Testing general notification...")
    result1 = send_general_notification("🧪 **System Test**: Testing the centralized Discord notification system")
    print(f"General notification result: {result1}")
      # Test scraper notification
    print("\n2. Testing data scraper notification...")
    result2 = send_scraper_notification("🧪 **Scraper Test**: Testing data scraper notifications")
    print(f"Scraper notification result: {result2}")
    
    # Test trainer notification
    print("\n3. Testing model trainer notification...")
    result3 = send_trainer_notification("🧪 **Trainer Test**: Testing model trainer notifications")
    print(f"Trainer notification result: {result3}")
    
    # Test trader notification
    print("\n4. Testing trader notification...")
    result4 = send_trader_notification("🧪 **Trader Test**: Testing trading bot notifications")
    print(f"Trader notification result: {result4}")
    
    print("\n✅ Notification testing completed!")
    print("Check your Discord channels to see if the notifications were received.")

if __name__ == "__main__":
    test_notifications()

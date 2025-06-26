#!/usr/bin/env python3
"""
Test script to verify the health check endpoint is working
"""

import requests
import json
import sys

def test_health_check(url="http://localhost:8000/health"):
    """Test the health check endpoint"""
    try:
        print(f"🧪 Testing health check at: {url}")
        
        response = requests.get(url, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ JSON Response:")
                print(json.dumps(data, indent=2))
                
                # Check required fields
                required_fields = ["status", "service", "health_server"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                    return False
                else:
                    print(f"✅ All required fields present")
                    return True
                    
            except json.JSONDecodeError:
                print(f"❌ Response is not valid JSON: {response.text}")
                return False
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/health"
    success = test_health_check(url)
    sys.exit(0 if success else 1)

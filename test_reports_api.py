#!/usr/bin/env python3
"""
Test Reports API Endpoint
"""

import requests
import json

def test_reports_api():
    """Test the reports overview API endpoint"""
    try:
        url = "http://localhost:5000/api/reports/overview"
        print(f"🧪 Testing Reports API: {url}")
        
        response = requests.get(url, timeout=10)
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS - Reports data received")
            
            # Show key metrics
            if 'summary' in data:
                summary = data['summary']
                print(f"\n📈 REPORTS SUMMARY:")
                print(f"   Total Sessions: {summary.get('total_sessions', 'N/A')}")
                print(f"   Total Events: {summary.get('total_hotspots', 'N/A')}")
                print(f"   Avg Duration: {summary.get('avg_duration', 'N/A'):.1f}s")
                print(f"   Avg Events/Session: {summary.get('avg_events_per_session', 'N/A'):.1f}")
                
                if 'event_types' in data:
                    print(f"\n🎯 EVENT TYPES:")
                    for event_type, count in data['event_types'].items():
                        print(f"   {event_type}: {count}")
                        
                if 'risk_levels' in data:
                    print(f"\n⚠️ RISK LEVELS:")
                    for level, count in data['risk_levels'].items():
                        print(f"   {level.title()}: {count}")
                        
            return True
        else:
            print(f"❌ FAILED - HTTP {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    test_reports_api()
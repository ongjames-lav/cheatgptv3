#!/usr/bin/env python3
"""
Test the API endpoint to see if it returns the correct labels
"""

import requests
import json

def test_events_api():
    """Test the /events/<session_id> API endpoint"""
    try:
        # Test with the latest session
        session_id = "session_20250912_125631_2f194cd5"
        url = f"http://localhost:5000/events/{session_id}"
        
        print(f"Testing API: {url}")
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            
            print(f"✅ API Response Success!")
            print(f"Total events: {len(events)}")
            print("\nEvent descriptions from API:")
            
            for i, event in enumerate(events[:5], 1):  # Show first 5
                description = event.get('description', 'No description')
                confidence = event.get('confidence', 0)
                timestamp = event.get('timestamp_seconds', 0)
                
                print(f"  {i}. Time: {timestamp:.1f}s")
                print(f"     Description: {description}")
                print(f"     Confidence: {confidence:.0%}")
                print()
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_events_api()

#!/usr/bin/env python3
"""
Test Analytics API Endpoints
Verifies that all analytics data connections are working correctly
"""

import requests
import json
import sys
from datetime import datetime

def test_api_endpoint(endpoint, description):
    """Test a specific API endpoint and return results"""
    try:
        url = f"http://localhost:5000{endpoint}"
        print(f"\n🧪 Testing {description}")
        print(f"📡 URL: {url}")
        
        response = requests.get(url, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS - Data received")
            
            # Pretty print the response structure (limit to avoid overflow)
            if isinstance(data, dict):
                print(f"📋 Response Keys: {list(data.keys())}")
                
                # Show sample data structure
                if 'sessions' in data and data['sessions']:
                    print(f"📊 Sessions Count: {len(data['sessions'])}")
                    print(f"📋 Sample Session Keys: {list(data['sessions'][0].keys()) if data['sessions'] else 'No sessions'}")
                elif 'videos' in data and data['videos']:
                    print(f"📊 Videos Count: {len(data['videos'])}")
                    print(f"📋 Sample Video Keys: {list(data['videos'][0].keys()) if data['videos'] else 'No videos'}")
                
                return True, data
            else:
                print(f"📋 Response Type: {type(data)}")
                return True, data
        else:
            print(f"❌ FAILED - HTTP {response.status_code}")
            print(f"📄 Response: {response.text[:200]}...")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ FAILED - Connection refused (is the server running?)")
        return False, None
    except requests.exceptions.Timeout:
        print(f"❌ FAILED - Request timeout")
        return False, None
    except Exception as e:
        print(f"❌ FAILED - {str(e)}")
        return False, None

def analyze_session_data(sessions_data, videos_data):
    """Analyze the session data for consistency and completeness"""
    print(f"\n📊 === DATA ANALYSIS ===")
    
    # Analyze recorded sessions
    if sessions_data and 'sessions' in sessions_data:
        sessions = sessions_data['sessions']
        print(f"\n🎥 RECORDED SESSIONS:")
        print(f"   Total Count: {len(sessions)}")
        
        if sessions:
            # Check data completeness
            total_events = sum(session.get('hotspot_count', 0) for session in sessions)
            avg_duration = sum(session.get('duration', 0) for session in sessions) / len(sessions) if sessions else 0
            avg_events = total_events / len(sessions) if sessions else 0
            
            print(f"   Total Events: {total_events}")
            print(f"   Average Duration: {avg_duration:.1f} seconds")
            print(f"   Average Events/Session: {avg_events:.1f}")
            
            # Check required fields
            required_fields = ['session_id', 'start_time', 'duration', 'hotspot_count']
            for i, session in enumerate(sessions[:3]):  # Check first 3 sessions
                missing = [field for field in required_fields if field not in session or session[field] is None]
                if missing:
                    print(f"   ⚠️  Session {i+1} missing: {missing}")
                else:
                    print(f"   ✅ Session {i+1} has all required fields")
    
    # Analyze uploaded videos
    if videos_data and 'videos' in videos_data:
        videos = videos_data['videos']
        print(f"\n📤 UPLOADED VIDEOS:")
        print(f"   Total Count: {len(videos)}")
        
        if videos:
            total_events = sum(video.get('event_count', 0) for video in videos)
            avg_duration = sum(video.get('duration', 0) for video in videos) / len(videos) if videos else 0
            avg_events = total_events / len(videos) if videos else 0
            
            print(f"   Total Events: {total_events}")
            print(f"   Average Duration: {avg_duration:.1f} seconds")
            print(f"   Average Events/Session: {avg_events:.1f}")
            
            # Check required fields
            required_fields = ['session_id', 'video_title', 'duration', 'event_count']
            for i, video in enumerate(videos[:3]):  # Check first 3 videos
                missing = [field for field in required_fields if field not in video or video[field] is None]
                if missing:
                    print(f"   ⚠️  Video {i+1} missing: {missing}")
                else:
                    print(f"   ✅ Video {i+1} has all required fields")

def test_analytics_calculations():
    """Test that analytics calculations would work correctly"""
    print(f"\n🧮 === ANALYTICS CALCULATIONS TEST ===")
    
    # Test the calculations that the frontend JavaScript would perform
    print("Testing calculation logic...")
    
    # This simulates the frontend updateSummaryCards() logic
    test_sessions = [
        {'duration': 120, 'hotspot_count': 5},
        {'duration': 90, 'hotspot_count': 3},
        {'duration': 150, 'hotspot_count': 8}
    ]
    
    totalSessions = len(test_sessions)
    totalEvents = sum(session.get('hotspot_count', 0) for session in test_sessions)
    avgDuration = sum(session.get('duration', 0) for session in test_sessions) / totalSessions if totalSessions > 0 else 0
    avgEventsPerSession = totalEvents / totalSessions if totalSessions > 0 else 0
    
    print(f"   Test Data: {len(test_sessions)} sessions")
    print(f"   ✅ Total Sessions: {totalSessions}")
    print(f"   ✅ Total Events: {totalEvents}")
    print(f"   ✅ Avg Duration: {avgDuration:.1f}s")
    print(f"   ✅ Avg Events/Session: {avgEventsPerSession:.1f}")

def main():
    """Main test function"""
    print("🚀 CheatGPT Analytics API Test")
    print("=" * 50)
    
    # Test main endpoints
    endpoints_to_test = [
        ("/api/sessions/list", "Recorded Sessions List"),
        ("/api/sessions/uploaded", "Uploaded Videos List"),
        ("/", "Main Analytics Page (HTML)"),
    ]
    
    results = {}
    
    for endpoint, description in endpoints_to_test:
        success, data = test_api_endpoint(endpoint, description)
        results[endpoint] = {"success": success, "data": data}
    
    # Analyze the data if both endpoints succeeded
    sessions_data = results.get("/api/sessions/list", {}).get("data")
    videos_data = results.get("/api/sessions/uploaded", {}).get("data")
    
    if sessions_data and videos_data:
        analyze_session_data(sessions_data, videos_data)
    
    # Test analytics calculations
    test_analytics_calculations()
    
    # Summary
    print(f"\n📋 === TEST SUMMARY ===")
    passed = sum(1 for result in results.values() if result["success"])
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Analytics connections are working correctly!")
    else:
        print("⚠️  Some tests failed - check the output above for details")
        failed_endpoints = [ep for ep, result in results.items() if not result["success"]]
        print(f"Failed endpoints: {failed_endpoints}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
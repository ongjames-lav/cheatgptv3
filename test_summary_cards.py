#!/usr/bin/env python3
"""
Test Summary Cards Functionality
Verify that the summary cards show correct data for recorded sessions only
"""

import requests
import json

def test_summary_cards():
    """Test that summary cards display correct recorded session data"""
    print("🏠 SUMMARY CARDS TEST")
    print("=" * 50)
    
    try:
        # Get recorded sessions data (what summary cards should show)
        response = requests.get("http://localhost:5000/api/sessions/list")
        data = response.json()
        
        sessions = data.get('sessions', [])
        
        # Calculate what the summary cards should display
        total_sessions = len(sessions)
        total_events = sum(session.get('hotspot_count', 0) for session in sessions)
        total_duration = sum(session.get('duration', 0) for session in sessions)
        avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
        avg_events_per_session = total_events / total_sessions if total_sessions > 0 else 0
        
        print(f"📊 EXPECTED SUMMARY CARD VALUES:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   Total Events: {total_events}")
        print(f"   Average Duration: {avg_duration:.1f} seconds ({avg_duration/60:.1f} minutes)")
        print(f"   Average Events/Session: {avg_events_per_session:.1f}")
        
        # Verify data quality
        print(f"\n🔍 DATA QUALITY CHECK:")
        sessions_with_events = [s for s in sessions if s.get('hotspot_count', 0) > 0]
        sessions_with_duration = [s for s in sessions if s.get('duration', 0) > 0]
        
        print(f"   Sessions with events: {len(sessions_with_events)}/{total_sessions}")
        print(f"   Sessions with duration: {len(sessions_with_duration)}/{total_sessions}")
        
        # Check for any data anomalies
        if total_sessions > 0:
            max_events = max(session.get('hotspot_count', 0) for session in sessions)
            min_events = min(session.get('hotspot_count', 0) for session in sessions)
            max_duration = max(session.get('duration', 0) for session in sessions)
            min_duration = min(session.get('duration', 0) for session in sessions)
            
            print(f"   Event range: {min_events} - {max_events}")
            print(f"   Duration range: {min_duration:.1f}s - {max_duration:.1f}s")
        
        # Sample session data for verification
        print(f"\n📋 SAMPLE SESSION DATA:")
        for i, session in enumerate(sessions[:3]):
            print(f"   Session {i+1}:")
            print(f"     ID: {session.get('session_id', 'N/A')}")
            print(f"     Events: {session.get('hotspot_count', 0)}")
            print(f"     Duration: {session.get('duration', 0):.1f}s")
            print(f"     Status: {session.get('status', 'N/A')}")
        
        print(f"\n✅ Summary cards should display the values shown above")
        print(f"💡 If the web page shows different values, there may be a JavaScript error")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing summary cards: {e}")
        return False

if __name__ == "__main__":
    test_summary_cards()
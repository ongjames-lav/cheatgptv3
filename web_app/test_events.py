#!/usr/bin/env python3
"""
Test script to check the latest events and their formatting
"""

from db_manager import db

def test_latest_events():
    """Check the latest events in the database"""
    print("🔍 Testing Latest Events:")
    print("=" * 50)
    
    # Get latest session
    sessions = db.get_sessions(limit=1)
    if not sessions:
        print("❌ No sessions found")
        return
    
    session_id = sessions[0]['session_id']
    print(f"📊 Latest Session: {session_id}")
    
    # Get events
    events = db.get_session_events(session_id)
    print(f"📈 Total Events: {len(events)}")
    
    if not events:
        print("❌ No events found")
        return
    
    print("\n🎯 Recent Events:")
    print("-" * 30)
    
    # Show last 5 events
    for i, event in enumerate(events[-5:], 1):
        timestamp = event['timestamp_seconds']
        event_type = event['event_type']
        description = event['description']
        confidence = event['confidence']
        severity = event['severity']
        
        print(f"Event {i}:")
        print(f"  Time: {timestamp:.1f}s")
        print(f"  Type: {event_type}")
        print(f"  Description: {description}")
        print(f"  Confidence: {confidence:.0%}")
        print(f"  Severity: {severity}")
        print()
    
    # Check for unique event types
    unique_types = set()
    unique_descriptions = set()
    
    for event in events:
        unique_types.add(event['event_type'])
        unique_descriptions.add(event['description'])
    
    print(f"🔄 Unique Event Types ({len(unique_types)}):")
    for event_type in sorted(unique_types):
        print(f"  - {event_type}")
    
    print(f"\n📝 Unique Descriptions ({len(unique_descriptions)}):")
    for desc in sorted(unique_descriptions):
        print(f"  - {desc}")

if __name__ == "__main__":
    test_latest_events()

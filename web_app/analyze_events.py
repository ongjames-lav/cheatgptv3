import sys
sys.path.append('..')
from db_manager import DatabaseManager

db = DatabaseManager("cheatgpt_sessions.db")  # Use the correct database
sessions = db.get_sessions_with_details()

print("Detailed Session Analysis:")
print("=========================")

for i, session in enumerate(sessions[:5]):
    events = db.get_session_events(session['session_id'])
    duration = session.get('duration', 0)
    event_count = len(events)
    
    print(f"\nSession {i+1}: {session['session_id']}")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"  Total Events: {event_count}")
    
    if duration > 0:
        events_per_min = event_count / (duration / 60)
        print(f"  Events/min: {events_per_min:.2f}")
    else:
        events_per_min = 0
        print(f"  Events/min: 0 (no duration)")
    
    # Show event types and confidence levels
    if events:
        print("  Event breakdown:")
        event_types = {}
        for event in events[:10]:  # Show first 10 events
            event_type = event.get('event_type', 'unknown')
            confidence = event.get('confidence', 0)
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(confidence)
            
        for event_type, confidences in event_types.items():
            avg_confidence = sum(confidences) / len(confidences)
            print(f"    {event_type}: {len(confidences)} events, avg confidence: {avg_confidence:.2f}")
    
    # Calculate with our current algorithm
    if duration > 0 and event_count > 0:
        if events_per_min <= 0.5:
            suspicious = round(events_per_min * 50)
        elif events_per_min <= 1.0:
            suspicious = round(25 + (events_per_min - 0.5) * 50)
        elif events_per_min <= 2.0:
            suspicious = round(50 + (events_per_min - 1.0) * 25)
        else:
            suspicious = round(75 + min(25, (events_per_min - 2.0) * 12.5))
        suspicious = max(0, min(100, suspicious))
        print(f"  Current algorithm result: {suspicious}% suspicious")
    else:
        print(f"  Current algorithm result: 0% suspicious")

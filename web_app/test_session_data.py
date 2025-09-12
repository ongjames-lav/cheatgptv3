import sys
sys.path.append('..')
from db_manager import DatabaseManager

db = DatabaseManager()
sessions = db.get_sessions_with_details()

print("Session Data Check:")
print("==================")

for i, session in enumerate(sessions[:3]):
    events = db.get_session_events(session['session_id'])
    duration = session.get('duration', 0)
    event_count = len(events)
    
    if duration > 0:
        events_per_min = event_count / (duration / 60)
    else:
        events_per_min = 0
        
    print(f"Session {i+1}: {session['session_id']}")
    print(f"  Duration: {duration} seconds")
    print(f"  Events: {event_count}")
    print(f"  Events/min: {events_per_min:.2f}")
    print(f"  Should show: {0 if event_count == 0 else 'calculated'}% suspicious")
    print()

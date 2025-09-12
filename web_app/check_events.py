from db_manager import db

# Check the actual event types stored in the database
session_id = 'session_20250912_032135_26520088'
events = db.get_session_events(session_id)

print(f'Session: {session_id}')
print(f'Total events: {len(events)}')
print()

if events:
    print('Sample events from database:')
    for i, event in enumerate(events[:5]):  # Show first 5 events
        event_type = event.get('event_type', 'None')
        confidence = event.get('confidence', 0)
        timestamp = event.get('timestamp_seconds', 0)
        print(f'  Event {i+1}:')
        print(f'    Event Type: "{event_type}"')
        print(f'    Confidence: {confidence}')
        print(f'    Timestamp: {timestamp}s')
        print()
        
    # Check unique event types
    unique_types = set(event.get('event_type', 'None') for event in events)
    print(f'Unique event types in database: {list(unique_types)}')
else:
    print('No events found in database')

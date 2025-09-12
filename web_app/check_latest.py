#!/usr/bin/env python3
"""
Quick test to check the latest session events
"""

from db_manager import db

def check_latest_session():
    sessions = db.get_sessions(limit=1)
    if sessions:
        session_id = sessions[0]['session_id'] 
        print(f'Latest session: {session_id}')
        events = db.get_session_events(session_id)
        print(f'Total events: {len(events)}')
        if events:
            print('\nAll events in latest session:')
            for i, event in enumerate(events, 1):
                event_type = event['event_type']
                description = event['description'] 
                confidence = event['confidence']
                timestamp = event['timestamp_seconds']
                print(f'  {i}. Time: {timestamp:.1f}s')
                print(f'     Type: {event_type}')
                print(f'     Description: {description}')
                print(f'     Confidence: {confidence:.0%}')
                print()
    else:
        print('No sessions found')

if __name__ == "__main__":
    check_latest_session()

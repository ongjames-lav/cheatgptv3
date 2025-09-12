import sqlite3
import os

# Check both databases to understand the actual data
db_path = "cheatgpt.db"

print("=== DEBUGGING ANALYTICS DATA ===")

# Check main database
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n1. SAMPLE SESSIONS DATA:")
    cursor.execute("SELECT session_id, start_timestamp, end_timestamp, cam_id FROM sessions LIMIT 5")
    sessions = cursor.fetchall()
    for session in sessions:
        session_id, start_ts, end_ts, cam_id = session
        duration = (end_ts - start_ts) if end_ts else 0
        print(f"   Session {session_id}: Duration={duration:.1f}s, Cam={cam_id}")
    
    print("\n2. SAMPLE EVENTS DATA:")
    cursor.execute("SELECT timestamp, cam_id, event_type, confidence FROM events LIMIT 10")
    events = cursor.fetchall()
    for event in events:
        print(f"   Event: Time={event[0]}, Cam={event[1]}, Type={event[2]}, Confidence={event[3]}")
    
    print("\n3. EVENTS COUNT BY CAM_ID (which should match session cam_id):")
    cursor.execute("""
        SELECT cam_id, COUNT(*) as event_count
        FROM events 
        GROUP BY cam_id
        LIMIT 10
    """)
    events_by_cam = cursor.fetchall()
    for event_group in events_by_cam:
        print(f"   Cam {event_group[0]}: {event_group[1]} events")
    
    print("\n4. TRYING TO MATCH SESSIONS WITH EVENTS BY CAM_ID:")
    cursor.execute("""
        SELECT s.session_id, s.start_timestamp, s.end_timestamp, s.cam_id, COUNT(e.cam_id) as event_count
        FROM sessions s 
        LEFT JOIN events e ON s.cam_id = e.cam_id 
            AND e.timestamp >= s.start_timestamp 
            AND e.timestamp <= s.end_timestamp
        GROUP BY s.session_id, s.start_timestamp, s.end_timestamp, s.cam_id
        LIMIT 10
    """)
    sessions_with_events = cursor.fetchall()
    for session in sessions_with_events:
        session_id, start_ts, end_ts, cam_id, event_count = session
        duration = (end_ts - start_ts) if end_ts else 0
        events_per_min = (event_count / (duration / 60.0)) if duration > 0 else 0
        print(f"   Session {session_id}: {event_count} events in {duration:.1f}s = {events_per_min:.2f}/min")
    
    conn.close()
else:
    print(f"Main database {db_path} not found")

print("\n=== END DEBUG ===")

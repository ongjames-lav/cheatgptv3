import sys, os
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager
import time

db = DatabaseManager()

print("=" * 60)
print("ANALYZING SESSION ORDER BY TIMESTAMP")
print("=" * 60)

# Get all sessions (default limit 50)
all_sessions_50 = db.get_sessions_with_details(limit=50)
all_sessions_100 = db.get_sessions_with_details(limit=100)
all_sessions_150 = db.get_sessions_with_details(limit=150)

print(f"\nWith limit=50: {len(all_sessions_50)} sessions")
print(f"With limit=100: {len(all_sessions_100)} sessions")  
print(f"With limit=150: {len(all_sessions_150)} sessions")

# Count types in limit=100
recorded_100 = [s for s in all_sessions_100 if s.get('session_type') != 'uploaded']
uploaded_100 = [s for s in all_sessions_100 if s.get('session_type') == 'uploaded']

print(f"\nIn first 100 sessions:")
print(f"  Recorded: {len(recorded_100)}")
print(f"  Uploaded: {len(uploaded_100)}")

# Show newest sessions
print(f"\nNewest 10 sessions (by start_time):")
for i, s in enumerate(all_sessions_100[:10]):
    session_type = s.get('session_type', 'recorded')
    start_ts = s.get('start_time', 0)
    from datetime import datetime
    date_str = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S') if start_ts else 'N/A'
    print(f"  {i+1}. {s['session_id']} ({session_type}) - {date_str}")

print("\n" + "=" * 60)

import sys, os
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager

db = DatabaseManager()

print("=" * 60)
print("ANALYZING SESSION COUNTS")
print("=" * 60)

# Get all sessions
all_sessions = db.get_sessions_with_details(limit=150)

# Count by type
recorded = [s for s in all_sessions if s.get('session_type') == 'recorded' or s.get('session_type') is None]
uploaded = [s for s in all_sessions if s.get('session_type') == 'uploaded']

print(f"\nTotal sessions in database: {len(all_sessions)}")
print(f"  Recorded: {len(recorded)}")
print(f"  Uploaded: {len(uploaded)}")

print(f"\nFirst 10 uploaded sessions:")
for s in uploaded[:10]:
    print(f"  - {s['session_id']}: {s.get('video_title')}")
    print(f"    Path: {s.get('video_path')}")
    print(f"    Type: {s.get('session_type')}, Status: {s.get('status')}")

print("\n" + "=" * 60)

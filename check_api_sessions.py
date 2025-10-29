import sys, os
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager

db = DatabaseManager()

# Check the two sessions showing in the API
sessions_to_check = ['single_1761065641', 'single_1761065596']

print("=" * 60)
print("CHECKING SESSIONS FROM API RESPONSE")
print("=" * 60)

for session_id in sessions_to_check:
    print(f"\nSession: {session_id}")
    s = db.get_session(session_id)
    if s:
        print(f"  Found in database:")
        print(f"    video_path: {s.get('video_path')}")
        print(f"    video_title: {s.get('video_title')}")
        print(f"    session_type: {s.get('session_type')}")
        print(f"    status: {s.get('status')}")
    else:
        print(f"  NOT FOUND in database!")

print("\n" + "=" * 60)

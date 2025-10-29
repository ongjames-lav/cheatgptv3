import sys, os
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager
from pathlib import Path

db = DatabaseManager()

print("=" * 60)
print("CHECKING UPLOADED SESSIONS IN DATABASE")
print("=" * 60)

# Get all sessions with session_type = 'uploaded'
sessions = db.get_sessions_with_details(limit=100)
uploaded = [s for s in sessions if s.get('session_type') == 'uploaded']

print(f"\nFound {len(uploaded)} uploaded sessions:\n")

for s in uploaded[:10]:
    session_id = s['session_id']
    video_path = s.get('video_path', 'N/A')
    
    print(f"Session: {session_id}")
    print(f"  Video Path in DB: {video_path}")
    
    # Check if file exists at that path
    if video_path and video_path != 'N/A':
        # Try relative to web_app
        path1 = Path('web_app') / video_path
        # Try relative to project root
        path2 = Path(video_path)
        
        print(f"  Checking: web_app/{video_path}")
        print(f"    Exists: {path1.exists()}")
        print(f"  Checking: {video_path}")
        print(f"    Exists: {path2.exists()}")
        
        # Find actual file
        results_path = Path('web_app/results') / session_id / f'processed_{session_id}.mp4'
        print(f"  Expected at: {results_path}")
        print(f"    Exists: {results_path.exists()}")
    
    print()

print("=" * 60)

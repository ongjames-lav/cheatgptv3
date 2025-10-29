import sys, os
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager
from pathlib import Path
import time

print("=" * 60)
print("CHECKING UPLOADED VIDEOS IN RESULTS FOLDER")
print("=" * 60)

results_dir = Path('web_app/results')
if results_dir.exists():
    video_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('single_')]
    print(f"\nFound {len(video_dirs)} uploaded video directories:")
    
    for video_dir in sorted(video_dirs):
        video_files = list(video_dir.glob('processed_*.mp4'))
        if video_files:
            video_file = video_files[0]
            size_mb = video_file.stat().st_size / 1024 / 1024
            print(f"  - {video_dir.name}: {video_file.name} ({size_mb:.1f} MB)")
        else:
            print(f"  - {video_dir.name}: NO VIDEO FILE")
else:
    print(f"❌ Results directory not found at: {results_dir}")

print("\n" + "=" * 60)
print("CHECKING DATABASE")
print("=" * 60)

db = DatabaseManager()

# Check what's in database
cursor = db.conn.cursor()
cursor.execute("SELECT session_id, video_path, session_type FROM sessions WHERE session_type = 'uploaded' ORDER BY session_id")
db_sessions = cursor.fetchall()

print(f"\nUpload videos in database: {len(db_sessions)}")
for row in db_sessions:
    print(f"  - {row[0]}: {row[1]} (type: {row[2]})")

db.conn.close()

print("\n" + "=" * 60)

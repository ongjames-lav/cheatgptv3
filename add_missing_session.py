import sys
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager
from pathlib import Path

db = DatabaseManager()

print("=" * 60)
print("ADDING MISSING UPLOADED VIDEO TO DATABASE")
print("=" * 60)

session_id = 'single_1761068335'
video_dir = Path(f'web_app/results/{session_id}')
video_file = video_dir / f'processed_{session_id}.mp4'

if video_file.exists():
    print(f"\n✅ Found video: {video_file}")
    
    # Get file info
    file_stats = video_file.stat()
    start_ts = file_stats.st_ctime
    file_size = file_stats.st_size
    
    # Try to get duration
    duration = 0
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_file))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0:
                duration = frame_count / fps
            cap.release()
    except:
        pass
    
    end_ts = start_ts + duration if duration > 0 else start_ts
    
    # Make path relative to web_app
    relative_path = f"results\\{session_id}\\processed_{session_id}.mp4"
    
    try:
        # Check if already exists
        existing = db.get_session(session_id)
        if existing:
            print(f"⏭️  Session already exists in database")
        else:
            # Add to database
            db.create_uploaded_session(
                session_id=session_id,
                video_path=relative_path,
                video_title=f"processed_{session_id}.mp4",
                start_ts=start_ts,
                end_ts=end_ts,
                duration=duration,
                metadata={
                    'source': 'manual_add',
                    'file_size': file_size
                }
            )
            print(f"✅ Added {session_id} to database")
            print(f"   Video path: {relative_path}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   File size: {file_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print(f"❌ Video file not found: {video_file}")

print("\n" + "=" * 60)

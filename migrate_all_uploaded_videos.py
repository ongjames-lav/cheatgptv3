import sys, os
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager
from pathlib import Path
import time
import json

# Initialize database - this will create/connect to web_app/cheatgpt_sessions.db
db = DatabaseManager()

print("=" * 70)
print("MIGRATING ALL PROCESSED VIDEOS TO WEBAPP DATABASE")
print("=" * 70)

# Find all processed videos in results folder
results_dir = Path('web_app/results')
uploaded_count = 0
skipped_count = 0
error_count = 0

if results_dir.exists():
    print(f"\n📂 Scanning: {results_dir}")
    
    # Get all session directories
    session_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('single_')])
    print(f"Found {len(session_dirs)} session directories\n")
    
    for idx, session_dir in enumerate(session_dirs, 1):
        session_id = session_dir.name
        
        # Find processed video file
        video_files = list(session_dir.glob('processed_*.mp4'))
        if not video_files:
            print(f"[{idx:2d}/{len(session_dirs)}] ⚠️  {session_id}: No video found")
            continue
        
        video_path = video_files[0]
        video_title = video_path.name
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        
        # Check if session already exists
        existing = db.get_session(session_id)
        if existing:
            print(f"[{idx:2d}/{len(session_dirs)}] ⏭️  {session_id}: Already in database ({file_size_mb:.1f} MB)")
            skipped_count += 1
            continue
        
        # Get file stats for timestamp
        file_stats = video_path.stat()
        start_ts = file_stats.st_ctime
        
        # Try to get duration from video metadata
        duration = 0
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps > 0:
                    duration = frame_count / fps
                cap.release()
        except Exception as e:
            # If video metadata fails, estimate from file size
            pass
        
        end_ts = start_ts + duration if duration > 0 else start_ts
        
        # Make path relative to web_app directory
        relative_path = f"results\\{session_id}\\{video_title}"
        
        # Create session in webapp database
        try:
            session_pk = db.create_uploaded_session(
                session_id=session_id,
                video_path=relative_path,
                video_title=video_title,
                start_ts=start_ts,
                end_ts=end_ts,
                duration=duration,
                metadata={
                    'source': 'migration',
                    'original_filename': video_title,
                    'migrated_at': time.time()
                }
            )
            
            print(f"[{idx:2d}/{len(session_dirs)}] ✅ {session_id}: {file_size_mb:6.1f} MB")
            uploaded_count += 1
            
        except Exception as e:
            print(f"[{idx:2d}/{len(session_dirs)}] ❌ {session_id}: {str(e)}")
            error_count += 1
else:
    print("❌ Results directory not found")

print("\n" + "=" * 70)
print(f"MIGRATION SUMMARY")
print(f"  ✅ Migrated: {uploaded_count}")
print(f"  ⏭️  Skipped:  {skipped_count}")
print(f"  ❌ Errors:   {error_count}")
print(f"  📊 Total:    {uploaded_count + skipped_count + error_count}")
print("=" * 70)

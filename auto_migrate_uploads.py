import sys, os
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager
from pathlib import Path
import time

db = DatabaseManager()

print("=" * 60)
print("AUTO-MIGRATING ALL UPLOADED VIDEOS")
print("=" * 60)

# Find all processed videos in results folder
results_dir = Path('web_app/results')
uploaded_count = 0
skipped_count = 0

if results_dir.exists():
    for session_dir in sorted(results_dir.iterdir()):
        if session_dir.is_dir() and session_dir.name.startswith('single_'):
            session_id = session_dir.name
            
            # Check if session already exists
            existing = db.get_session(session_id)
            if existing:
                skipped_count += 1
                continue
            
            # Find processed video file
            video_files = list(session_dir.glob('processed_*.mp4'))
            if not video_files:
                print(f"⚠️  No video found for {session_id}")
                continue
            
            video_path = video_files[0]
            video_title = video_path.name
            
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
            except:
                pass
            
            end_ts = start_ts + duration if duration > 0 else start_ts
            
            # Create session in webapp database
            try:
                relative_path = f"results\\{session_id}\\{video_file.name}"
                
                db.create_uploaded_session(
                    session_id=session_id,
                    video_path=relative_path,
                    video_title=video_title,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    duration=duration,
                    metadata={
                        'source': 'auto_migration',
                        'original_filename': video_title,
                        'migrated_at': time.time()
                    }
                )
                
                print(f"✅ Migrated {session_id}")
                uploaded_count += 1
                
            except Exception as e:
                print(f"❌ Error migrating {session_id}: {e}")
else:
    print("❌ Results directory not found")

print("\n" + "=" * 60)
print(f"MIGRATION COMPLETE")
print(f"  ✅ Migrated: {uploaded_count}")
print(f"  ⏭️  Skipped (already in DB): {skipped_count}")
print("=" * 60)

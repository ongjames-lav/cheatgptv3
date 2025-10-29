import sys, os
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager
from pathlib import Path
import time
import json

# Ensure we're using the correct database path (web_app directory)
db_path = Path('web_app/cheatgpt_sessions.db').absolute()
db = DatabaseManager(str(db_path))

print("=" * 60)
print("MIGRATING EXISTING PROCESSED VIDEOS TO WEBAPP DATABASE")
print("=" * 60)

# Find all processed videos in results folder
results_dir = Path('web_app/results')
uploaded_count = 0
skipped_count = 0

if results_dir.exists():
    for session_dir in results_dir.iterdir():
        if session_dir.is_dir() and session_dir.name.startswith('single_'):
            session_id = session_dir.name
            
            # Check if session already exists
            existing = db.get_session(session_id)
            if existing:
                print(f"⏭️  Skipping {session_id} - already in database")
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
            
            # Check for events/reports
            reports_dir = Path('web_app/uploads/reports')
            event_count = 0
            
            # Try to read event count from JSON report
            json_report = reports_dir / 'json' / f'report_{session_id}.json'
            if json_report.exists():
                try:
                    with open(json_report, 'r') as f:
                        report_data = json.load(f)
                        event_count = report_data.get('total_events', 0)
                except:
                    pass
            
            # Create session in webapp database
            try:
                # Make path relative to web_app directory with correct format
                relative_path = f"results\\{session_id}\\{video_title}"
                
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
                
                print(f"✅ Migrated {session_id}")
                print(f"   Video: {relative_path}")
                print(f"   Duration: {duration:.1f}s")
                print(f"   Events: {event_count}")
                
                uploaded_count += 1
                
            except Exception as e:
                print(f"❌ Error migrating {session_id}: {e}")
else:
    print("❌ Results directory not found")

print("\n" + "=" * 60)
print(f"MIGRATION COMPLETE")
print(f"  ✅ Migrated: {uploaded_count}")
print(f"  ⏭️  Skipped: {skipped_count}")
print("=" * 60)

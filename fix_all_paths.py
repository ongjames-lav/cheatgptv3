import sqlite3
from pathlib import Path

db_path = 'web_app/cheatgpt_sessions.db'
results_dir = Path('web_app/results')

print("=" * 60)
print("FIXING ALL UPLOADED VIDEO PATHS")
print("=" * 60)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all uploaded sessions
cursor.execute("SELECT session_id, video_path FROM sessions WHERE session_type = 'uploaded'")
uploaded = cursor.fetchall()

print(f"\nFound {len(uploaded)} uploaded sessions in database")

fixed_count = 0
for session_id, old_path in uploaded:
    # Check if file exists in results folder
    expected_video_file = None
    session_results_dir = results_dir / session_id
    
    if session_results_dir.exists():
        video_files = list(session_results_dir.glob('processed_*.mp4'))
        if video_files:
            expected_video_file = video_files[0]
    
    if expected_video_file:
        # Construct correct path
        correct_path = f"results\\{session_id}\\{expected_video_file.name}"
        
        if old_path != correct_path:
            print(f"\n📹 {session_id}")
            print(f"   Old: {old_path}")
            print(f"   New: {correct_path}")
            
            # Update database
            cursor.execute(
                "UPDATE sessions SET video_path = ?, video_title = ? WHERE session_id = ?",
                (correct_path, expected_video_file.name, session_id)
            )
            fixed_count += 1
            print(f"   ✅ Updated")
        else:
            print(f"✓ {session_id} - path already correct")
    else:
        print(f"⚠️  {session_id} - No video file found in results folder")

conn.commit()
conn.close()

print("\n" + "=" * 60)
print(f"FIXED {fixed_count} sessions")
print("=" * 60)

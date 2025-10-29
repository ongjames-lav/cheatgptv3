import sqlite3
from pathlib import Path

db_path = 'web_app/cheatgpt_sessions.db'

print("=" * 60)
print("FIXING UPLOADED VIDEO PATHS IN DATABASE")
print("=" * 60)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all uploaded sessions
cursor.execute("SELECT session_id, video_path FROM sessions WHERE session_type = 'uploaded'")
uploaded = cursor.fetchall()

print(f"\nFound {len(uploaded)} uploaded sessions")

fixed_count = 0
for session_id, old_path in uploaded:
    print(f"\n📹 Session: {session_id}")
    print(f"   Old path: {old_path}")
    
    # Check if file exists at results location
    expected_path = f"results\\{session_id}\\processed_{session_id}.mp4"
    full_path = Path(f"web_app/{expected_path}")
    
    if full_path.exists():
        print(f"   ✅ Found file at: {expected_path}")
        
        # Update database
        cursor.execute(
            "UPDATE sessions SET video_path = ?, video_title = ? WHERE session_id = ?",
            (expected_path, f"processed_{session_id}.mp4", session_id)
        )
        fixed_count += 1
        print(f"   ✅ Updated database")
    else:
        print(f"   ❌ File not found at: {full_path}")

conn.commit()
conn.close()

print("\n" + "=" * 60)
print(f"FIXED {fixed_count} sessions")
print("=" * 60)

import sqlite3
from pathlib import Path
import requests
import os

session_id = 'single_1761069614'

print("=" * 70)
print(f"DEBUGGING VIDEO PLAYBACK: {session_id}")
print("=" * 70)

# Test 1: Check file system
print("\n✓ TEST 1: FILE SYSTEM")
print("-" * 70)

video_dir = Path(f'web_app/results/{session_id}')
print(f"Directory: {video_dir}")
print(f"Exists: {video_dir.exists()}")

if video_dir.exists():
    video_files = list(video_dir.glob('*.mp4'))
    print(f"Video files found: {len(video_files)}")
    for f in video_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.1f} MB)")
else:
    print(f"❌ Directory does not exist!")

# Test 2: Check database
print("\n✓ TEST 2: DATABASE")
print("-" * 70)

db_path = 'web_app/cheatgpt_sessions.db'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
session = cursor.fetchone()

if session:
    print(f"✓ Session found in database")
    print(f"  session_id: {session['session_id']}")
    print(f"  video_path: {session['video_path']}")
    print(f"  video_title: {session['video_title']}")
    print(f"  session_type: {session['session_type']}")
    print(f"  status: {session['status']}")
    
    # Verify the path exists
    full_path = Path(f"web_app/{session['video_path']}")
    print(f"  Full path: {full_path}")
    print(f"  Path exists: {full_path.exists()}")
else:
    print(f"❌ Session NOT in database")

conn.close()

# Test 3: Check API
print("\n✓ TEST 3: API ENDPOINT")
print("-" * 70)

try:
    response = requests.get(f'http://localhost:5000/api/sessions/list', timeout=5)
    if response.status_code == 200:
        data = response.json()
        sessions = data.get('sessions', [])
        found = next((s for s in sessions if s['session_id'] == session_id), None)
        
        if found:
            print(f"✓ Session found in API")
            print(f"  video_path: {found['video_path']}")
            print(f"  session_type: {found['session_type']}")
        else:
            print(f"❌ Session NOT in API response")
            print(f"   Total sessions in API: {len(sessions)}")
    else:
        print(f"❌ API error: {response.status_code}")
except Exception as e:
    print(f"❌ Cannot reach API: {e}")

# Test 4: Check playback endpoint
print("\n✓ TEST 4: PLAYBACK ENDPOINT")
print("-" * 70)

try:
    response = requests.get(f'http://localhost:5000/playback/{session_id}', timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Content-Length: {response.headers.get('Content-Length')}")
    
    if response.status_code == 200:
        print("✓ Playback working!")
    else:
        print(f"❌ Error: {response.text[:200]}")
except Exception as e:
    print(f"❌ Cannot reach playback: {e}")
    print("Make sure Flask is running!")

print("\n" + "=" * 70)

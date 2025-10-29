#!/usr/bin/env python
"""
Comprehensive test to verify uploaded videos are properly integrated
"""
import requests
import sqlite3
from pathlib import Path

print("=" * 70)
print("UPLOADED VIDEO INTEGRATION TEST")
print("=" * 70)

# Test 1: Check database for uploaded videos
print("\n✓ Test 1: Checking database for uploaded videos")
db_path = 'web_app/cheatgpt_sessions.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM sessions WHERE session_type = 'uploaded'")
uploaded_count = cursor.fetchone()[0]
print(f"  Uploaded videos in database: {uploaded_count}")

cursor.execute("SELECT session_id, video_path FROM sessions WHERE session_type = 'uploaded' LIMIT 3")
for row in cursor.fetchall():
    print(f"    - {row[0]}: {row[1]}")

conn.close()

# Test 2: Check API endpoint
print("\n✓ Test 2: Testing /api/sessions/list endpoint")
try:
    response = requests.get('http://localhost:5000/api/sessions/list', timeout=5)
    if response.status_code == 200:
        data = response.json()
        uploaded_sessions = [s for s in data['sessions'] if s.get('session_type') == 'uploaded']
        print(f"  API returned {len(uploaded_sessions)} uploaded videos")
        if uploaded_sessions:
            print(f"  First uploaded video: {uploaded_sessions[0]['session_id']}")
    else:
        print(f"  ❌ API error: {response.status_code}")
except Exception as e:
    print(f"  ❌ Connection error: {e}")

# Test 3: Check playback
print("\n✓ Test 3: Testing video playback")
test_session = 'single_1761068335'
try:
    response = requests.head(f'http://localhost:5000/playback/{test_session}', timeout=5)
    if response.status_code == 200:
        print(f"  ✅ Video {test_session} is playable (HTTP 200)")
    else:
        print(f"  ❌ Playback error: {response.status_code}")
except Exception as e:
    print(f"  ⚠️  Cannot test playback: {e}")

# Test 4: Check files exist
print("\n✓ Test 4: Checking video files in results folder")
results_dir = Path('web_app/results')
if results_dir.exists():
    video_count = len(list(results_dir.glob('single_*/processed_*.mp4')))
    print(f"  Found {video_count} processed video files")
else:
    print(f"  ❌ Results folder not found")

print("\n" + "=" * 70)
print("✅ All tests completed!")
print("=" * 70)

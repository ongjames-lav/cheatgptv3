#!/usr/bin/env python3
"""
Quick check script - verify uploaded video integration is working
Run this before and after uploading videos to confirm everything is working
"""

import sqlite3
from pathlib import Path
import requests
import sys

def check_database():
    """Check database status"""
    print("\n📊 DATABASE STATUS")
    print("─" * 60)
    
    db_path = 'web_app/cheatgpt_sessions.db'
    if not Path(db_path).exists():
        print("❌ Database not found!")
        return False
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check uploaded videos
    cursor.execute("SELECT COUNT(*) as count FROM sessions WHERE session_type = 'uploaded'")
    uploaded_count = cursor.fetchone()['count']
    
    # Check recorded videos
    cursor.execute("SELECT COUNT(*) as count FROM sessions WHERE session_type IS NULL OR session_type = 'recorded'")
    recorded_count = cursor.fetchone()['count']
    
    print(f"✓ Recorded videos: {recorded_count}")
    print(f"✓ Uploaded videos: {uploaded_count}")
    
    if uploaded_count > 0:
        cursor.execute("""
            SELECT session_id, video_path FROM sessions 
            WHERE session_type = 'uploaded'
            ORDER BY session_id DESC
            LIMIT 3
        """)
        print("\nLatest uploaded videos:")
        for row in cursor.fetchall():
            print(f"  • {row['session_id']}")
    
    conn.close()
    return True

def check_results_folder():
    """Check results folder"""
    print("\n📁 RESULTS FOLDER STATUS")
    print("─" * 60)
    
    results_dir = Path('web_app/results')
    if not results_dir.exists():
        print("❌ Results folder not found!")
        return False
    
    session_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('single_')]
    video_files = []
    for session_dir in session_dirs:
        videos = list(session_dir.glob('processed_*.mp4'))
        video_files.extend(videos)
    
    print(f"✓ Session directories: {len(session_dirs)}")
    print(f"✓ Processed video files: {len(video_files)}")
    
    return True

def check_api():
    """Check API endpoints"""
    print("\n🌐 API STATUS")
    print("─" * 60)
    
    try:
        # Check /api/sessions/list
        response = requests.get('http://localhost:5000/api/sessions/list', timeout=5)
        if response.status_code == 200:
            data = response.json()
            uploaded = [s for s in data['sessions'] if s.get('session_type') == 'uploaded']
            recorded = [s for s in data['sessions'] if s.get('session_type') != 'uploaded']
            print(f"✓ /api/sessions/list: {response.status_code}")
            print(f"  ├─ Recorded: {len(recorded)}")
            print(f"  └─ Uploaded: {len(uploaded)}")
            return True
        else:
            print(f"❌ /api/sessions/list: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach Flask app: {e}")
        print("   Start Flask with: python web_app/app.py")
        return False

def check_playback():
    """Check playback endpoint"""
    print("\n▶️  PLAYBACK STATUS")
    print("─" * 60)
    
    db_path = 'web_app/cheatgpt_sessions.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get first uploaded video
    cursor.execute("""
        SELECT session_id FROM sessions 
        WHERE session_type = 'uploaded'
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("ℹ️  No uploaded videos to test")
        return True
    
    session_id = row['session_id']
    
    try:
        response = requests.head(f'http://localhost:5000/playback/{session_id}', timeout=5)
        if response.status_code == 200:
            print(f"✓ /playback/{session_id}: {response.status_code}")
            size = response.headers.get('Content-Length')
            if size:
                mb = int(size) / 1024 / 1024
                print(f"  └─ Size: {mb:.1f} MB")
            return True
        else:
            print(f"❌ /playback/{session_id}: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach playback: {e}")
        return False

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("UPLOADED VIDEO INTEGRATION - QUICK CHECK")
    print("=" * 60)
    
    results = {
        "Database": check_database(),
        "Results Folder": check_results_folder(),
        "API": check_api(),
        "Playback": check_playback()
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = all(results.values())
    
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All systems operational!")
        print("Ready to upload and play videos.")
    else:
        print("⚠️  Some systems not operational")
        print("Check errors above and restart Flask if needed:")
        print("  python web_app/app.py")
    print("=" * 60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Clean up orphaned sessions (sessions without video files)
"""

import sqlite3
import os
from pathlib import Path

def clean_orphaned_sessions():
    """Remove sessions that don't have corresponding video files"""
    
    # Get all video files
    videos_dir = Path("videos")
    video_files = list(videos_dir.glob("session_*.mp4"))
    
    # Extract session IDs from video files
    video_session_ids = set()
    for video_file in video_files:
        filename = video_file.name
        # Parse the double session_ format: session_session_20250918_090142_a4dbddf8_...
        if filename.startswith("session_session_"):
            parts = filename.replace('.mp4', '').split('_')
            if len(parts) >= 4:
                # Reconstruct proper session ID
                session_id = f"session_{parts[2]}_{parts[3]}_{parts[4]}"
                video_session_ids.add(session_id)
    
    print(f"📁 Found {len(video_session_ids)} sessions with video files")
    
    # Get all database sessions
    try:
        with sqlite3.connect("cheatgpt_sessions.db") as conn:
            cursor = conn.execute("SELECT session_id FROM sessions")
            db_session_ids = {row[0] for row in cursor.fetchall()}
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return
    
    print(f"📊 Found {len(db_session_ids)} sessions in database")
    
    # Find orphaned sessions (in DB but no video)
    orphaned_sessions = db_session_ids - video_session_ids
    
    if orphaned_sessions:
        print(f"\n🗑️ Found {len(orphaned_sessions)} orphaned sessions:")
        for session_id in sorted(orphaned_sessions):
            print(f"  - {session_id}")
        
        # Ask for confirmation
        response = input(f"\nDelete these {len(orphaned_sessions)} orphaned sessions? (y/N): ")
        
        if response.lower() == 'y':
            try:
                with sqlite3.connect("cheatgpt_sessions.db") as conn:
                    for session_id in orphaned_sessions:
                        # Delete hotspots first
                        cursor = conn.execute("DELETE FROM hotspots WHERE session_id = ?", (session_id,))
                        hotspots_deleted = cursor.rowcount
                        
                        # Delete session
                        cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                        sessions_deleted = cursor.rowcount
                        
                        print(f"🗑️ Deleted {session_id} ({hotspots_deleted} hotspots)")
                    
                    conn.commit()
                    print(f"\n✅ Successfully deleted {len(orphaned_sessions)} orphaned sessions")
            except Exception as e:
                print(f"❌ Error deleting sessions: {e}")
        else:
            print("❌ Deletion cancelled")
    else:
        print("\n✅ No orphaned sessions found!")
    
    # Final summary
    try:
        with sqlite3.connect("cheatgpt_sessions.db") as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            final_db_count = cursor.fetchone()[0]
    except:
        final_db_count = 0
    
    print(f"\n📊 FINAL STATE:")
    print(f"   Database sessions: {final_db_count}")
    print(f"   Video files: {len(video_session_ids)}")
    
    if final_db_count == len(video_session_ids):
        print("🎉 Perfect match! Web interface should now show all sessions.")
    else:
        print("⚠️ Still some discrepancy exists.")

if __name__ == "__main__":
    clean_orphaned_sessions()
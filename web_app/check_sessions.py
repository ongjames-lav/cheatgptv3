#!/usr/bin/env python3
"""
Script to check session database vs video files discrepancy
"""

import sqlite3
import os
import glob
from pathlib import Path

def check_database_sessions():
    """Check sessions in the database"""
    db_path = "cheatgpt_sessions.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return []
    
    try:
        with sqlite3.connect(db_path) as conn:
            # First check what columns exist
            cursor = conn.execute("PRAGMA table_info(sessions)")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"📋 Database columns: {columns}")
            print()
            
            # Query based on available columns
            if 'video_title' in columns:
                query = """
                    SELECT session_id, video_title, start_ts, end_ts, duration, status, created_at
                    FROM sessions 
                    ORDER BY created_at DESC
                """
            else:
                query = """
                    SELECT session_id, start_ts, end_ts, duration, status, created_at
                    FROM sessions 
                    ORDER BY created_at DESC
                """
            
            cursor = conn.execute(query)
            
            sessions = []
            print("📊 SESSIONS IN DATABASE:")
            print("-" * 80)
            
            for i, row in enumerate(cursor.fetchall(), 1):
                if 'video_title' in columns:
                    session_id, video_title, start_ts, end_ts, duration, status, created_at = row
                    print(f"    Title: {video_title or 'None'}")
                else:
                    session_id, start_ts, end_ts, duration, status, created_at = row
                    print(f"    Title: (column not exists)")
                
                sessions.append(session_id)
                print(f"{i:2d}. {session_id}")
                print(f"    Status: {status}")
                print(f"    Created: {created_at}")
                print(f"    Duration: {duration}s" if duration else "    Duration: None")
                print()
            
            print(f"📈 Total sessions in database: {len(sessions)}")
            return sessions
            
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return []

def check_video_files():
    """Check video files in the videos directory"""
    videos_dir = Path("videos")
    
    if not videos_dir.exists():
        print(f"❌ Videos directory not found: {videos_dir}")
        return []
    
    # Get all video files
    video_files = list(videos_dir.glob("*.mp4"))
    
    print("🎬 VIDEO FILES IN DIRECTORY:")
    print("-" * 80)
    
    session_files = []
    for i, video_file in enumerate(sorted(video_files), 1):
        filename = video_file.name
        print(f"{i:2d}. {filename}")
        
        # Extract session ID from filename
        if filename.startswith("session_"):
            # Try to extract session ID pattern
            parts = filename.replace(".mp4", "").split("_")
            if len(parts) >= 4:
                # session_session_20250918_090142_a4dbddf8_20250918_090147
                # Expected pattern: session_session_YYYYMMDD_HHMMSS_hash_...
                session_id = "_".join(parts[:4])  # session_session_20250918_090142_a4dbddf8
                session_files.append(session_id)
                print(f"    Extracted session ID: {session_id}")
        print()
    
    print(f"📈 Total video files: {len(video_files)}")
    print(f"📈 Session video files: {len(session_files)}")
    
    return session_files

def compare_sessions_and_files():
    """Compare database sessions with video files"""
    print("=" * 80)
    print("🔍 CHECKING SESSION/VIDEO FILE DISCREPANCY")
    print("=" * 80)
    print()
    
    db_sessions = check_database_sessions()
    print()
    
    video_sessions = check_video_files()
    print()
    
    print("🔄 COMPARISON RESULTS:")
    print("-" * 80)
    
    # Sessions in DB but no video file
    db_only = set(db_sessions) - set(video_sessions)
    if db_only:
        print("📊 Sessions in database but NO video file:")
        for session in sorted(db_only):
            print(f"  - {session}")
        print()
    
    # Video files but no DB session
    video_only = set(video_sessions) - set(db_sessions)
    if video_only:
        print("🎬 Video files but NO database session:")
        for session in sorted(video_only):
            print(f"  - {session}")
        print()
    
    # Sessions in both
    both = set(db_sessions) & set(video_sessions)
    if both:
        print("✅ Sessions in BOTH database and video files:")
        for session in sorted(both):
            print(f"  - {session}")
        print()
    
    print("📊 SUMMARY:")
    print(f"  Database sessions: {len(db_sessions)}")
    print(f"  Video files: {len(video_sessions)}")
    print(f"  In both: {len(both)}")
    print(f"  DB only: {len(db_only)}")
    print(f"  Video only: {len(video_only)}")
    
    if len(db_sessions) != len(video_sessions):
        print()
        print("⚠️  DISCREPANCY DETECTED!")
        print("   This explains why the web interface shows fewer sessions than video files.")
        print("   The web interface shows database sessions, not video files directly.")

if __name__ == "__main__":
    compare_sessions_and_files()
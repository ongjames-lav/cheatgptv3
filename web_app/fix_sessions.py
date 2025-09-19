#!/usr/bin/env python3
"""
Script to fix session database vs video files discrepancy
"""

import sqlite3
import os
import glob
import time
from pathlib import Path
from datetime import datetime

def parse_video_filename(filename):
    """Parse video filename to extract session information"""
    # Remove .mp4 extension
    base_name = filename.replace('.mp4', '')
    
    # Example: session_session_20250918_090142_a4dbddf8_20250918_090147
    parts = base_name.split('_')
    
    if len(parts) >= 6 and parts[0] == 'session' and parts[1] == 'session':
        # Extract components
        date_part = parts[2]  # 20250918
        time_part = parts[3]  # 090142  
        hash_part = parts[4]  # a4dbddf8
        end_date = parts[5] if len(parts) > 5 else date_part  # 20250918
        end_time = parts[6] if len(parts) > 6 else time_part  # 090147
        
        # Construct proper session_id (remove double session_)
        session_id = f"session_{date_part}_{time_part}_{hash_part}"
        
        # Parse start time
        start_datetime_str = f"{date_part}_{time_part}"
        try:
            start_dt = datetime.strptime(start_datetime_str, "%Y%m%d_%H%M%S")
            start_timestamp = start_dt.timestamp()
        except:
            start_timestamp = time.time()
        
        # Parse end time  
        end_datetime_str = f"{end_date}_{end_time}"
        try:
            end_dt = datetime.strptime(end_datetime_str, "%Y%m%d_%H%M%S")
            end_timestamp = end_dt.timestamp()
        except:
            end_timestamp = start_timestamp + 60  # Default 1 minute
        
        return {
            'session_id': session_id,
            'original_filename': filename,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'duration': end_timestamp - start_timestamp,
            'video_path': f"videos/{filename}"
        }
    
    return None

def get_video_file_info():
    """Get information about all video files"""
    videos_dir = Path("videos")
    video_files = list(videos_dir.glob("*.mp4"))
    
    parsed_videos = []
    for video_file in video_files:
        filename = video_file.name
        if filename.startswith("session_") and filename != "tobol.mp4":
            info = parse_video_filename(filename)
            if info:
                parsed_videos.append(info)
    
    return parsed_videos

def get_existing_sessions():
    """Get existing sessions from database"""
    existing_sessions = set()
    
    try:
        with sqlite3.connect("cheatgpt_sessions.db") as conn:
            cursor = conn.execute("SELECT session_id FROM sessions")
            existing_sessions = {row[0] for row in cursor.fetchall()}
    except Exception as e:
        print(f"Error reading existing sessions: {e}")
    
    return existing_sessions

def import_missing_sessions():
    """Import video sessions that are missing from database"""
    video_infos = get_video_file_info()
    existing_sessions = get_existing_sessions()
    
    print("🔄 IMPORTING MISSING SESSIONS:")
    print("-" * 60)
    
    imported_count = 0
    
    try:
        with sqlite3.connect("cheatgpt_sessions.db") as conn:
            for video_info in video_infos:
                session_id = video_info['session_id']
                
                if session_id not in existing_sessions:
                    # Import this session
                    print(f"📥 Importing: {session_id}")
                    print(f"   File: {video_info['original_filename']}")
                    print(f"   Duration: {video_info['duration']:.1f}s")
                    
                    cursor = conn.execute("""
                        INSERT INTO sessions (session_id, video_path, start_ts, end_ts, duration, status, created_at)
                        VALUES (?, ?, ?, ?, ?, 'completed', datetime('now'))
                    """, (
                        session_id,
                        video_info['video_path'],
                        video_info['start_timestamp'],
                        video_info['end_timestamp'],
                        video_info['duration']
                    ))
                    
                    imported_count += 1
                else:
                    print(f"✅ Already exists: {session_id}")
            
            conn.commit()
            
    except Exception as e:
        print(f"❌ Error importing sessions: {e}")
        return 0
    
    print()
    print(f"📊 IMPORT SUMMARY:")
    print(f"   Sessions imported: {imported_count}")
    print(f"   Sessions skipped: {len(video_infos) - imported_count}")
    
    return imported_count

def verify_fix():
    """Verify that the fix worked"""
    print("\n" + "="*60)
    print("🔍 VERIFYING FIX:")
    print("="*60)
    
    # Count database sessions
    try:
        with sqlite3.connect("cheatgpt_sessions.db") as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            db_count = cursor.fetchone()[0]
    except:
        db_count = 0
    
    # Count video files
    video_infos = get_video_file_info()
    video_count = len(video_infos)
    
    print(f"📊 Database sessions: {db_count}")
    print(f"🎬 Video sessions: {video_count}")
    
    if db_count == video_count:
        print("✅ SUCCESS: Session counts now match!")
    else:
        print("⚠️  Still some discrepancy exists")
    
    return db_count, video_count

if __name__ == "__main__":
    print("🔧 FIXING SESSION/VIDEO DISCREPANCY")
    print("="*60)
    
    # Show current state
    video_infos = get_video_file_info()
    existing_sessions = get_existing_sessions()
    
    print(f"📊 Current state:")
    print(f"   Database sessions: {len(existing_sessions)}")
    print(f"   Video files: {len(video_infos)}")
    print()
    
    # Import missing sessions
    imported = import_missing_sessions()
    
    # Verify fix
    verify_fix()
    
    print()
    print("🎉 Fix process completed!")
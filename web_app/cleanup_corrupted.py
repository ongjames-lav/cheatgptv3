#!/usr/bin/env python3
"""
Remove corrupted video files and their database entries
"""

import sqlite3
import os
from pathlib import Path

def remove_corrupted_files():
    """Remove corrupted video files and their database sessions"""
    
    corrupted_files = [
        "session_session_20250918_121845_21d6c772_20250918_121848.mp4",
        "session_session_20250918_130305_4c8908a7_20250918_130308.mp4"
    ]
    
    # Extract session IDs
    corrupted_session_ids = []
    for filename in corrupted_files:
        parts = filename.replace('.mp4', '').split('_')
        if len(parts) >= 4:
            session_id = f"session_{parts[2]}_{parts[3]}_{parts[4]}"
            corrupted_session_ids.append(session_id)
    
    print("🗑️ CORRUPTED FILES TO REMOVE:")
    for i, (filename, session_id) in enumerate(zip(corrupted_files, corrupted_session_ids), 1):
        print(f"  {i}. {filename}")
        print(f"     Session ID: {session_id}")
    
    response = input(f"\nRemove these {len(corrupted_files)} corrupted files and their database entries? (y/N): ")
    
    if response.lower() == 'y':
        # Remove files
        videos_dir = Path("videos")
        files_removed = 0
        
        for filename in corrupted_files:
            file_path = videos_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"🗑️ Removed file: {filename}")
                    files_removed += 1
                except Exception as e:
                    print(f"❌ Error removing {filename}: {e}")
            else:
                print(f"⚠️ File not found: {filename}")
        
        # Remove database entries
        sessions_removed = 0
        try:
            with sqlite3.connect("cheatgpt_sessions.db") as conn:
                for session_id in corrupted_session_ids:
                    # Delete hotspots first
                    cursor = conn.execute("DELETE FROM hotspots WHERE session_id = ?", (session_id,))
                    hotspots_deleted = cursor.rowcount
                    
                    # Delete session
                    cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                    if cursor.rowcount > 0:
                        print(f"🗑️ Removed session: {session_id} ({hotspots_deleted} hotspots)")
                        sessions_removed += 1
                
                conn.commit()
        except Exception as e:
            print(f"❌ Error removing database entries: {e}")
        
        print(f"\n✅ CLEANUP COMPLETE:")
        print(f"   Files removed: {files_removed}")
        print(f"   Sessions removed: {sessions_removed}")
        
        # Final count
        try:
            with sqlite3.connect("cheatgpt_sessions.db") as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                final_count = cursor.fetchone()[0]
            print(f"   Final session count: {final_count}")
        except:
            pass
    
    else:
        print("❌ Cleanup cancelled")

if __name__ == "__main__":
    print("🧹 CORRUPTED FILE CLEANUP")
    print("=" * 50)
    remove_corrupted_files()
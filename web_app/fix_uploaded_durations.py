"""
Fix duration for uploaded video sessions that have processing time instead of actual video duration
"""
import sqlite3
import json

def fix_uploaded_sessions():
    """Update duration field for uploaded sessions to use video_metadata.duration"""
    db_path = 'cheatgpt_sessions.db'
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all uploaded sessions
    cursor.execute("""
        SELECT session_id, duration, metadata, start_ts, end_ts 
        FROM sessions 
        WHERE session_type = 'uploaded' OR status = 'uploaded'
    """)
    
    sessions = cursor.fetchall()
    print(f"Found {len(sessions)} uploaded sessions")
    
    fixed_count = 0
    for session_id, db_duration, metadata_str, start_ts, end_ts in sessions:
        try:
            # Parse metadata
            metadata = json.loads(metadata_str) if metadata_str else {}
            video_metadata = metadata.get('video_metadata', {})
            actual_duration = video_metadata.get('duration')
            
            if actual_duration and abs(actual_duration - db_duration) > 1.0:  # More than 1 second difference
                print(f"\n📝 Fixing {session_id}:")
                print(f"   Current DB duration: {db_duration:.1f}s ({db_duration/60:.1f} min)")
                print(f"   Actual video duration: {actual_duration:.1f}s ({actual_duration/60:.1f} min)")
                print(f"   Processing time was: {end_ts - start_ts:.1f}s")
                
                # Update the duration
                cursor.execute("""
                    UPDATE sessions 
                    SET duration = ? 
                    WHERE session_id = ?
                """, (actual_duration, session_id))
                
                fixed_count += 1
                print(f"   ✅ Fixed!")
            else:
                print(f"✓ {session_id} already has correct duration")
                
        except Exception as e:
            print(f"❌ Error processing {session_id}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"Summary: Fixed {fixed_count} out of {len(sessions)} uploaded sessions")
    print(f"{'='*60}")

if __name__ == '__main__':
    fix_uploaded_sessions()

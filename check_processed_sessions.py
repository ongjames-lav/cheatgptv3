#!/usr/bin/env python3
"""
Check processed video sessions in database
"""

import sys
import os
import sqlite3

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_app.db_manager import DatabaseManager

def check_processed_sessions():
    """Check if processed video sessions exist in database"""
    print("=== CHECKING PROCESSED SESSIONS IN DATABASE ===\n")
    
    # Available processed videos
    processed_sessions = ['single_1759079854', 'single_1759237842', 'single_1759238054']
    
    db = DatabaseManager()
    
    for session_id in processed_sessions:
        print(f"🔍 Checking session: {session_id}")
        
        # Check if session exists in database
        session = db.get_session(session_id)
        if session:
            print(f"  ✅ Session found in database")
            print(f"     Status: {session.get('status', 'N/A')}")
            print(f"     Video Path: {session.get('video_path', 'N/A')}")
            print(f"     Duration: {session.get('duration', 'N/A')} seconds")
            print(f"     Start Time: {session.get('start_ts', 'N/A')}")
            print(f"     End Time: {session.get('end_ts', 'N/A')}")
            
            # Check for events/hotspots
            events = db.get_session_events(session_id)
            if events:
                print(f"     📊 Events: {len(events)} hotspots found")
            else:
                print(f"     ⚠️  No events/hotspots found")
        else:
            print(f"  ❌ Session NOT found in database")
            print(f"     This session needs to be added to the database")
        
        print()
    
    return processed_sessions

def add_missing_sessions():
    """Add missing processed video sessions to database"""
    print("=== ADDING MISSING SESSIONS ===\n")
    
    processed_sessions = ['single_1759079854', 'single_1759237842', 'single_1759238054']
    results_dir = "results"
    
    db = DatabaseManager()
    
    for session_id in processed_sessions:
        session = db.get_session(session_id)
        if not session:
            print(f"➕ Adding session: {session_id}")
            
            # Extract timestamp from session_id
            timestamp = int(session_id.replace('single_', ''))
            
            # Create session with processed video path
            processed_video_path = f"{results_dir}/{session_id}/processed_{session_id}.mp4"
            
            try:
                # Create session entry
                session_pk = db.create_session(
                    session_id=session_id,
                    start_ts=timestamp,
                    metadata={
                        'type': 'processed_upload',
                        'has_bounding_boxes': True,
                        'source': 'upload_processing'
                    }
                )
                
                # Update session with processed video path and completion
                db.end_session(
                    session_id=session_id,
                    end_ts=timestamp + 60,  # Assume 60 seconds for now
                    video_path=processed_video_path
                )
                
                print(f"  ✅ Session {session_id} added successfully")
                
            except Exception as e:
                print(f"  ❌ Error adding session {session_id}: {e}")
        else:
            print(f"  ✅ Session {session_id} already exists")

if __name__ == "__main__":
    # Check current state
    check_processed_sessions()
    
    # Add missing sessions
    add_missing_sessions()
    
    print("\n=== VERIFICATION ===")
    check_processed_sessions()
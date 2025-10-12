#!/usr/bin/env python3
"""
Script to manually add an uploaded video session to the database
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cheatgpt.db.db_manager import DBManager

def add_manual_session():
    """Add the missing session manually"""
    
    session_id = "processed_single_1760023563"
    
    # Initialize database
    db = DBManager()
    
    print(f"Adding manual session: {session_id}")
    
    # Check if session already exists
    existing_session = db.get_session_info(session_id)
    if existing_session:
        print("Session already exists!")
        return
    
    # Get the processed video path from downloads
    downloads_folder = Path("C:/Users/admin/Downloads")
    video_files = list(downloads_folder.glob("processed_*.mp4"))
    
    if not video_files:
        print("No processed video found in downloads folder")
        return
    
    # Find the most recent one
    video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    processed_video_path = str(video_files[0])
    
    print(f"Found processed video: {processed_video_path}")
    
    # Create the session entry
    success = db.create_uploaded_video_session(
        session_id=session_id,
        original_filename="uploaded_video.mp4",
        video_path=processed_video_path,  # Use the processed video as the original path for now
        video_metadata={
            'title': 'Manually Added Uploaded Video',
            'processed_manually': True
        }
    )
    
    if success:
        print("✅ Session created successfully!")
        
        # Update with processed results
        results = {
            'processing_completed': True,
            'processed_video_path': processed_video_path,
            'manual_entry': True,
            'total_events': 0  # Will be updated when events are added
        }
        
        db.update_processed_video_results(session_id, results)
        print("✅ Processing results updated!")
        
        # Add some dummy events for testing
        dummy_events = [
            {
                'event_type': 'phone_detection',
                'confidence': 0.85,
                'timestamp': time.time() - 100,
                'bbox': '100,150,200,250'
            },
            {
                'event_type': 'head_turning', 
                'confidence': 0.72,
                'timestamp': time.time() - 50,
                'bbox': '150,100,250,200'
            }
        ]
        
        db.store_uploaded_video_events(session_id, dummy_events)
        print("✅ Sample events added!")
        
        # Verify the session was created
        session = db.get_session_info(session_id)
        if session:
            print(f"✅ Session verified: {session['session_id']}")
            print(f"   Status: {session.get('status')}")
            print(f"   Type: {session.get('session_type')}")
            print(f"   Video path: {session.get('processed_video_path')}")
        else:
            print("❌ Session verification failed")
    else:
        print("❌ Failed to create session")

if __name__ == "__main__":
    add_manual_session()
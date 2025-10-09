#!/usr/bin/env python3

from cheatgpt.db.db_manager import DBManager

def main():
    db = DBManager()
    sessions = db.get_uploaded_video_sessions()
    
    print(f"Found {len(sessions)} uploaded video sessions:")
    for session in sessions:
        print(f"- {session['session_id']}: {session.get('status', 'unknown')}")
        print(f"  Processed video path: {session.get('processed_video_path', 'no path')}")
        print(f"  Original filename: {session.get('original_filename', 'no filename')}")
        print(f"  Total events: {session.get('total_events', 0)}")
        print()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from cheatgpt.db.db_manager import DBManager

def main():
    db = DBManager()
    
    # Check for the specific session
    session_id = "single_1760028860"
    session_info = db.get_session_info(session_id)
    
    if session_info:
        print(f"✅ Found session {session_id}:")
        print(f"  Status: {session_info.get('status', 'unknown')}")
        print(f"  Processed video path: {session_info.get('processed_video_path', 'no path')}")
        print(f"  Total events: {session_info.get('total_events', 0)}")
        print(f"  Session type: {session_info.get('session_type', 'unknown')}")
        print(f"  Original filename: {session_info.get('original_filename', 'no filename')}")
    else:
        print(f"❌ Session {session_id} not found")
    
    # Also check all recent sessions
    print("\n📋 All recent sessions:")
    sessions = db.get_all_sessions(limit=5)
    for session in sessions:
        print(f"- {session['session_id']}: {session.get('status', 'unknown')} - Events: {session.get('total_events', 0)}")

if __name__ == "__main__":
    main()
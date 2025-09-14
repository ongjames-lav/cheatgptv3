#!/usr/bin/env python3
"""
Quick script to check sessions and their hotspot counts in the database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_manager import DatabaseManager

def check_sessions():
    db = DatabaseManager()
    
    print("Checking sessions and hotspot counts...")
    sessions = db.get_sessions_with_details(limit=10)
    
    if not sessions:
        print("No sessions found in database.")
        return
    
    print(f"Found {len(sessions)} sessions:")
    for session in sessions:
        print(f"  Session ID: {session['session_id']}")
        print(f"    Hotspot Count: {session['hotspot_count']}")
        print(f"    Duration: {session['duration']} seconds")
        print(f"    Start Time: {session['start_time']}")
        print(f"    Status: {session['status']}")
        print()

if __name__ == "__main__":
    check_sessions()
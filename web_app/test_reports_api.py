#!/usr/bin/env python3
"""
Test script to check what data the reports API is returning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_manager import DatabaseManager
import json
import time
from datetime import datetime, timedelta

def test_reports_api():
    db = DatabaseManager()
    
    # Simulate the same logic as in app.py for /api/reports/overview
    date_range = '30'  # 30 days
    
    # Calculate date filter
    now = time.time()
    if date_range == 'all':
        start_time = 0
    else:
        days = int(date_range)
        start_time = now - (days * 24 * 60 * 60)
    
    print(f"Current time: {now}")
    print(f"Start time filter: {start_time}")
    print(f"Date range: {date_range} days")
    print()
    
    # Get all sessions in date range
    all_sessions = db.get_sessions_with_details(limit=1000)
    filtered_sessions = [s for s in all_sessions if s['start_time'] >= start_time]
    
    print(f"Total sessions: {len(all_sessions)}")
    print(f"Filtered sessions: {len(filtered_sessions)}")
    print()
    
    # Show first few sessions that would be returned
    recent_sessions = filtered_sessions[:5]  # Top 5 recent sessions
    
    print("Recent sessions that would be returned to frontend:")
    for session in recent_sessions:
        print(f"  Session ID: {session['session_id']}")
        print(f"    hotspot_count: {session['hotspot_count']}")
        print(f"    duration: {session['duration']}")
        print(f"    start_time: {session['start_time']}")
        print(f"    metadata: {session.get('metadata', {})}")
        print()

if __name__ == "__main__":
    test_reports_api()
#!/usr/bin/env python3
"""
Debug script to check if the latest session single_1760145724 is properly stored and accessible
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheatgpt.db.db_manager import DBManager

def main():
    print("🔍 Debugging latest session single_1760145724...")
    print("=" * 60)
    
    # Initialize database manager
    db = DBManager()
    
    # Check if the specific session exists
    session_id = "single_1760145724"
    print(f"📋 Checking for session: {session_id}")
    
    # Get session info
    session_info = None
    try:
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions WHERE session_id = ?
        """, (session_id,))
        result = cursor.fetchone()
        if result:
            # Convert to dict
            columns = [description[0] for description in cursor.description]
            session_info = dict(zip(columns, result))
    except Exception as e:
        print(f"❌ Error querying session: {e}")
    
    if session_info:
        print("✅ Session found in database!")
        print(f"   Session data: {session_info}")
        print()
        
        # Check required fields
        required_fields = ['session_id', 'video_path', 'status', 'session_type']
        missing_fields = []
        for field in required_fields:
            if field not in session_info or session_info[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"⚠️  Missing or null fields: {missing_fields}")
        else:
            print("✅ All required fields present")
    else:
        print("❌ Session NOT found in database!")
        print()
    
    # Get all uploaded videos to see what's available
    print("📋 All uploaded videos in database:")
    try:
        all_uploaded = db.get_uploaded_video_sessions()
        print(f"   Total uploaded videos: {len(all_uploaded)}")
        
        # Show last 5 sessions for comparison
        print("\n📊 Last 5 uploaded sessions:")
        for i, video in enumerate(all_uploaded[:5]):  # Already sorted by created_at DESC
            print(f"   {i+1}. Session: {video.get('session_id', 'N/A')}")
            print(f"      Status: {video.get('status', 'N/A')}")
            print(f"      Type: {video.get('session_type', 'N/A')}")
            print(f"      Events: {video.get('total_events', 'N/A')}")
            print(f"      Path: {video.get('video_path', 'N/A')}")
            print(f"      Processed Path: {video.get('processed_video_path', 'N/A')}")
            print()
            
        # Check if our target session is in the list
        target_found = False
        for video in all_uploaded:
            if video.get('session_id') == session_id:
                target_found = True
                break
        
        if target_found:
            print(f"✅ Session {session_id} found in uploaded videos list")
        else:
            print(f"❌ Session {session_id} NOT found in uploaded videos list")
            
    except Exception as e:
        print(f"❌ Error getting uploaded videos: {e}")
    
    # Test the database query directly
    print("\n🔍 Testing direct database query...")
    try:
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT session_id, status, session_type, total_events, video_path, processed_video_path, created_at
            FROM sessions 
            WHERE session_id = ?
        """, (session_id,))
        
        result = cursor.fetchone()
        if result:
            print("✅ Direct query found the session:")
            columns = ['session_id', 'status', 'session_type', 'total_events', 'video_path', 'processed_video_path', 'created_at']
            for i, col in enumerate(columns):
                print(f"   {col}: {result[i]}")
        else:
            print("❌ Direct query did not find the session")
            
        # Check all sessions in sessions table with session_type = 'uploaded'
        cursor.execute("""
            SELECT session_id, status, session_type, total_events, created_at 
            FROM sessions 
            WHERE session_type = 'uploaded'
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        all_sessions = cursor.fetchall()
        print(f"\n📊 Latest 10 uploaded sessions in sessions table:")
        for i, (sid, status, stype, events, created) in enumerate(all_sessions):
            print(f"   {i+1}. {sid} - {status} - {stype} - {events} events - {created}")
            
    except Exception as e:
        print(f"❌ Error with direct database query: {e}")

if __name__ == "__main__":
    main()
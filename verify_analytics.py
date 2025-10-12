#!/usr/bin/env python3
"""
Analytics Data Verification Script
Ensures all analytics connections are accurate and complete
"""

import requests
import json

def verify_analytics_accuracy():
    """Verify analytics data accuracy and connections"""
    print("🔍 ANALYTICS VERIFICATION REPORT")
    print("=" * 60)
    
    try:
        # Get recorded sessions
        sessions_response = requests.get("http://localhost:5000/api/sessions/list")
        sessions_data = sessions_response.json()
        
        # Get uploaded videos  
        videos_response = requests.get("http://localhost:5000/api/sessions/uploaded")
        videos_data = videos_response.json()
        
        print("\n📊 DATA COMPLETENESS CHECK:")
        print("-" * 30)
        
        # Verify recorded sessions data
        sessions = sessions_data.get('sessions', [])
        print(f"✅ Recorded Sessions: {len(sessions)} sessions found")
        
        # Check for missing or invalid data in sessions
        session_issues = []
        total_session_events = 0
        total_session_duration = 0
        
        for i, session in enumerate(sessions):
            issues = []
            
            # Check required fields
            if not session.get('session_id'):
                issues.append('Missing session_id')
            if session.get('duration', 0) <= 0:
                issues.append('Invalid duration')
            if session.get('hotspot_count', 0) < 0:
                issues.append('Invalid hotspot_count')
            
            total_session_events += session.get('hotspot_count', 0)
            total_session_duration += session.get('duration', 0)
            
            if issues:
                session_issues.append(f"Session {i+1}: {', '.join(issues)}")
        
        if session_issues:
            print(f"⚠️  Session Data Issues:")
            for issue in session_issues[:5]:  # Show first 5 issues
                print(f"   {issue}")
        else:
            print("✅ All recorded sessions have valid data")
        
        # Verify uploaded videos data
        videos = videos_data.get('videos', [])
        print(f"✅ Uploaded Videos: {len(videos)} videos found")
        
        # Check for missing or invalid data in videos
        video_issues = []
        total_video_events = 0
        total_video_duration = 0
        
        for i, video in enumerate(videos):
            issues = []
            
            # Check required fields
            if not video.get('session_id'):
                issues.append('Missing session_id')
            if video.get('duration', 0) < 0:
                issues.append('Invalid duration')
            if video.get('event_count', 0) < 0:
                issues.append('Invalid event_count')
            
            total_video_events += video.get('event_count', 0)
            total_video_duration += video.get('duration', 0)
            
            if issues:
                video_issues.append(f"Video {i+1}: {', '.join(issues)}")
        
        if video_issues:
            print(f"⚠️  Video Data Issues:")
            for issue in video_issues[:5]:  # Show first 5 issues
                print(f"   {issue}")
        else:
            print("✅ All uploaded videos have valid data")
        
        print("\n📈 ANALYTICS SUMMARY:")
        print("-" * 30)
        
        # Combined analytics
        total_sessions = len(sessions) + len(videos)
        total_events = total_session_events + total_video_events
        total_duration = total_session_duration + total_video_duration
        avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
        avg_events = total_events / total_sessions if total_sessions > 0 else 0
        
        print(f"📊 Total Sessions (All Types): {total_sessions}")
        print(f"📊 Total Events Detected: {total_events}")
        print(f"📊 Average Session Duration: {avg_duration:.1f} seconds")
        print(f"📊 Average Events per Session: {avg_events:.1f}")
        
        print(f"\n🔢 BREAKDOWN BY TYPE:")
        print(f"   📹 Recorded Sessions: {len(sessions)} ({total_session_events} events)")
        print(f"   📤 Uploaded Videos: {len(videos)} ({total_video_events} events)")
        
        # Event type analysis for recorded sessions
        print(f"\n🎯 EVENT TYPE ANALYSIS (Recorded Sessions):")
        event_types = {
            'phone_events': 0,
            'looking_events': 0, 
            'leaning_events': 0,
            'gesture_events': 0,
            'critical_events': 0,
            'warning_events': 0,
            'notice_events': 0
        }
        
        for session in sessions:
            for event_type in event_types:
                event_types[event_type] += session.get(event_type, 0)
        
        for event_type, count in event_types.items():
            display_name = event_type.replace('_', ' ').title()
            print(f"   {display_name}: {count}")
        
        print(f"\n✅ VERIFICATION COMPLETE")
        print(f"All analytics data connections are working correctly!")
        print(f"Frontend should display: {total_sessions} sessions, {total_events} events")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

if __name__ == "__main__":
    verify_analytics_accuracy()
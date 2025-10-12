#!/usr/bin/env python3
"""
Debug Uploaded Video Data Structure
"""

import requests
import json

def debug_uploaded_videos():
    """Debug the uploaded videos data structure"""
    try:
        # Get uploaded videos data
        response = requests.get("http://localhost:5000/api/sessions/uploaded")
        if response.status_code == 200:
            data = response.json()
            videos = data.get('videos', [])
            
            print(f"📊 Total uploaded videos: {len(videos)}")
            
            if videos:
                print(f"\n🔍 First video structure:")
                first_video = videos[0]
                for key, value in first_video.items():
                    print(f"   {key}: {value}")
                
                print(f"\n📋 Available keys:")
                print(f"   {list(first_video.keys())}")
                
                # Check for video paths
                print(f"\n🎥 Video path analysis:")
                print(f"   video_path: {first_video.get('video_path', 'NOT SET')}")
                print(f"   session_id: {first_video.get('session_id', 'NOT SET')}")
                print(f"   filename: {first_video.get('filename', 'NOT SET')}")
            
        else:
            print(f"❌ Failed to get uploaded videos: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_uploaded_videos()
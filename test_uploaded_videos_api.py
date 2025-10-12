#!/usr/bin/env python3
"""
Test script for processed videos API endpoint
"""

import requests
import json

def test_processed_videos_api():
    """Test the /api/sessions/uploaded endpoint (which now returns processed videos)"""
    try:
        # Make request to the processed videos API
        response = requests.get('http://localhost:5000/api/sessions/uploaded')
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success', False)}")
            
            videos = data.get('videos', [])
            print(f"Found {len(videos)} processed videos with bounding boxes")
            
            for i, video in enumerate(videos[:3]):  # Show first 3 videos
                print(f"\nProcessed Video {i+1}:")
                print(f"  Session ID: {video.get('session_id')}")
                print(f"  Filename: {video.get('filename')}")
                print(f"  Title: {video.get('video_title')}")
                print(f"  Size: {video.get('size')} bytes")
                print(f"  Processing Time: {video.get('formatted_time')}")
                print(f"  File Path: {video.get('file_path')}")
                print(f"  Duration: {video.get('duration')} seconds")
                print(f"  Events Count: {video.get('hotspot_count')}")
                print(f"  Has Bounding Boxes: {video.get('has_bounding_boxes')}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Flask server. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_processed_videos_api()
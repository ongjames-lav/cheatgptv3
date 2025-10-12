#!/usr/bin/env python3
"""
Debug script to check what the API is actually returning vs what's in the directories
"""

import os
import requests
import json

def check_directories():
    """Check what's actually in the directories"""
    print("=== DIRECTORY CONTENTS ===")
    
    # Check results directory
    results_dir = "results"
    print(f"\nResults directory: {results_dir}")
    print(f"Exists: {os.path.exists(results_dir)}")
    
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and item.startswith('single_'):
                print(f"  Session dir: {item}")
                for file in os.listdir(item_path):
                    print(f"    File: {file}")
                    if file.startswith('processed_'):
                        print(f"      ✅ PROCESSED VIDEO FOUND!")
    
    # Check uploads directory
    uploads_dir = "uploads"
    print(f"\nUploads directory: {uploads_dir}")
    print(f"Exists: {os.path.exists(uploads_dir)}")
    
    if os.path.exists(uploads_dir):
        count = 0
        for item in os.listdir(uploads_dir):
            item_path = os.path.join(uploads_dir, item)
            if os.path.isdir(item_path) and item.startswith('single_'):
                count += 1
                if count <= 3:  # Show first 3
                    print(f"  Session dir: {item}")
                    for file in os.listdir(item_path):
                        print(f"    File: {file}")
        print(f"  Total upload sessions: {count}")

def test_api():
    """Test the API endpoint"""
    print("\n=== API TEST ===")
    try:
        response = requests.get('http://localhost:5000/api/sessions/uploaded')
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            videos = data.get('videos', [])
            print(f"API returned {len(videos)} videos")
            
            for i, video in enumerate(videos[:3]):
                print(f"\nVideo {i+1}:")
                print(f"  Session ID: {video.get('session_id')}")
                print(f"  Filename: {video.get('filename')}")
                print(f"  File Path: {video.get('file_path')}")
                print(f"  Type: {video.get('type')}")
                
                # Check if the file path suggests it's from results or uploads
                file_path = video.get('file_path', '')
                if 'results/' in file_path:
                    print(f"  ✅ Correctly from results directory")
                elif 'single_' in file_path and not 'results/' in file_path:
                    print(f"  ❌ Appears to be from uploads directory")
                else:
                    print(f"  ❓ Unknown source")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Flask server. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_directories()
    test_api()
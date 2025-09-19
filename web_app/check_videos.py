#!/usr/bin/env python3
"""
Check for corrupted video files
"""

import cv2
import os
from pathlib import Path

def check_video_files():
    """Check all video files for corruption"""
    videos_dir = Path("videos")
    video_files = list(videos_dir.glob("session_*.mp4"))
    
    print(f"🎬 Checking {len(video_files)} video files for corruption...")
    print("-" * 60)
    
    corrupted_files = []
    working_files = []
    
    for video_file in sorted(video_files):
        try:
            cap = cv2.VideoCapture(str(video_file))
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # Get video info
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                print(f"✅ {video_file.name}")
                print(f"   Duration: {duration:.1f}s, Frames: {frame_count}, FPS: {fps:.1f}")
                working_files.append(video_file.name)
            else:
                print(f"❌ {video_file.name} - Cannot read frames")
                corrupted_files.append(video_file.name)
            
            cap.release()
            
        except Exception as e:
            print(f"❌ {video_file.name} - Error: {e}")
            corrupted_files.append(video_file.name)
        
        print()
    
    print("📊 SUMMARY:")
    print(f"   Working files: {len(working_files)}")
    print(f"   Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\n🚨 CORRUPTED FILES:")
        for file in corrupted_files:
            print(f"   - {file}")
        
        print(f"\n💡 These files might cause issues in the web interface.")
        print(f"   Consider removing them or re-recording those sessions.")

if __name__ == "__main__":
    check_video_files()
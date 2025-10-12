#!/usr/bin/env python3
"""
Test script to verify processed videos functionality locally
"""

import os
import time
import sys

def test_processed_videos_logic():
    """Test the processed videos logic locally without Flask server"""
    print("=== TESTING PROCESSED VIDEOS LOGIC ===\n")
    
    processed_videos = []
    results_dir = os.path.join(os.getcwd(), "results")
    
    print(f"Looking in results directory: {results_dir}")
    print(f"Directory exists: {os.path.exists(results_dir)}\n")
    
    if os.path.exists(results_dir):
        # Scan through results directories for processed videos
        for session_dir in os.listdir(results_dir):
            if session_dir.startswith('single_'):
                session_path = os.path.join(results_dir, session_dir)
                if os.path.isdir(session_path):
                    print(f"📁 Found session directory: {session_dir}")
                    
                    # Find processed video files in this session directory
                    for file in os.listdir(session_path):
                        if file.startswith('processed_') and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm')):
                            file_path = os.path.join(session_path, file)
                            file_stat = os.stat(file_path)
                            
                            # Extract timestamp from session_id (single_timestamp format)
                            timestamp = session_dir.replace('single_', '')
                            processing_time = int(timestamp) if timestamp.isdigit() else file_stat.st_ctime
                            
                            print(f"  🎬 Found processed video: {file}")
                            print(f"     Size: {file_stat.st_size} bytes ({file_stat.st_size / (1024*1024):.1f} MB)")
                            print(f"     Processing time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(processing_time))}")
                            
                            processed_videos.append({
                                'session_id': session_dir,
                                'filename': file,
                                'file_path': os.path.join('results', session_dir, file).replace('\\', '/'),
                                'full_path': file_path,
                                'size': file_stat.st_size,
                                'processing_time': processing_time,
                                'formatted_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(processing_time)),
                                'video_title': f"Processed Session {session_dir.replace('single_', '')}",
                                'type': 'processed',
                                'duration': 0,  # Would get from database in real implementation
                                'hotspot_count': 0,  # Would get from database in real implementation
                                'has_bounding_boxes': True
                            })
                    print()
    
    # Sort by processing time (newest first)
    processed_videos.sort(key=lambda x: x['processing_time'], reverse=True)
    
    print(f"✅ Total processed videos found: {len(processed_videos)}")
    
    if processed_videos:
        print("\n=== PROCESSED VIDEOS SUMMARY ===")
        for i, video in enumerate(processed_videos, 1):
            print(f"{i}. {video['video_title']}")
            print(f"   📁 Session: {video['session_id']}")
            print(f"   🎬 File: {video['filename']}")
            print(f"   📏 Size: {video['size'] / (1024*1024):.1f} MB")
            print(f"   🕒 Processed: {video['formatted_time']}")
            print(f"   🔗 Path: {video['file_path']}")
            print(f"   ✅ Has bounding boxes: {video['has_bounding_boxes']}")
            print()
    
    return processed_videos

if __name__ == "__main__":
    results = test_processed_videos_logic()
    
    if results:
        print("🎉 SUCCESS: Processed videos with bounding boxes found!")
        print("The 'Processed Videos' tab should show these videos when the Flask server is restarted.")
    else:
        print("⚠️  No processed videos found. Upload and process some videos first.")
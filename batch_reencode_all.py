#!/usr/bin/env python3
"""
Batch re-encode all incompatible videos to H.264
"""

import cv2
from pathlib import Path
import sys
import os

# Import the re-encoding function
sys.path.insert(0, str(Path(__file__).parent))
from reencode_video_h264 import reencode_video_h264

def check_video_codec(video_path):
    """Check if video uses browser-compatible codec"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, False
    
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()
    
    browser_compatible = fourcc_str in ['avc1', 'h264', 'H264', 'X264', 'x264']
    return fourcc_str, browser_compatible


def main():
    results_dir = Path("web_app/results")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("="*70)
    print("BATCH VIDEO RE-ENCODER - Convert All to H.264")
    print("="*70)
    print(f"Scanning: {results_dir}")
    print()
    
    # Find all processed videos that need re-encoding
    video_files = list(results_dir.glob("*/processed_*.mp4"))
    
    if not video_files:
        print("No processed videos found")
        sys.exit(0)
    
    print(f"Found {len(video_files)} processed videos")
    print("Checking codecs...")
    print()
    
    needs_reencoding = []
    
    for video_path in video_files:
        session_id = video_path.parent.name
        codec, is_compatible = check_video_codec(video_path)
        
        if codec and not is_compatible:
            needs_reencoding.append((session_id, video_path, codec))
            print(f"  ⚠️  {session_id}: {codec} (needs re-encoding)")
    
    if not needs_reencoding:
        print("✅ All videos are already browser-compatible!")
        sys.exit(0)
    
    print()
    print(f"Found {len(needs_reencoding)} videos that need re-encoding")
    print()
    
    # Ask for confirmation
    response = input(f"Re-encode {len(needs_reencoding)} videos? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Cancelled")
        sys.exit(0)
    
    print()
    print("="*70)
    print("RE-ENCODING VIDEOS")
    print("="*70)
    
    success_count = 0
    failed_count = 0
    
    for i, (session_id, video_path, codec) in enumerate(needs_reencoding, 1):
        print()
        print(f"[{i}/{len(needs_reencoding)}] {session_id}")
        
        try:
            if reencode_video_h264(video_path, backup=True):
                success_count += 1
            else:
                failed_count += 1
                print(f"   ❌ Failed to re-encode")
        except Exception as e:
            failed_count += 1
            print(f"   ❌ Error: {e}")
    
    print()
    print("="*70)
    print("BATCH RE-ENCODING COMPLETE")
    print("="*70)
    print(f"✅ Successfully re-encoded: {success_count}")
    print(f"❌ Failed: {failed_count}")
    
    if success_count > 0:
        print()
        print("🎉 Videos are now browser-compatible!")
        print("   Refresh your browser to see the changes")


if __name__ == '__main__':
    main()

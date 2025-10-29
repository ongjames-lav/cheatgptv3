#!/usr/bin/env python3
"""
Batch re-encode all videos in results folder to H.264 codec
This ensures all existing videos are browser-compatible
"""

import cv2
from pathlib import Path
import sys

def check_video_codec(video_path):
    """Check if video uses browser-compatible codec"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "Cannot open"
    
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
    print("BATCH VIDEO CODEC CHECKER")
    print("="*70)
    print(f"Scanning: {results_dir}")
    print()
    
    # Find all processed videos
    video_files = list(results_dir.glob("*/processed_*.mp4"))
    
    if not video_files:
        print("No processed videos found")
        sys.exit(0)
    
    print(f"Found {len(video_files)} processed videos")
    print()
    
    needs_reencoding = []
    compatible = []
    errors = []
    
    for video_path in video_files:
        session_id = video_path.parent.name
        codec, is_compatible = check_video_codec(video_path)
        
        if codec is None:
            errors.append((session_id, video_path, is_compatible))
            print(f"❌ {session_id}: {is_compatible}")
        elif is_compatible:
            compatible.append((session_id, codec))
            print(f"✅ {session_id}: {codec} (browser-compatible)")
        else:
            needs_reencoding.append((session_id, video_path, codec))
            print(f"⚠️  {session_id}: {codec} (needs re-encoding)")
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Browser-compatible: {len(compatible)}")
    print(f"⚠️  Needs re-encoding: {len(needs_reencoding)}")
    print(f"❌ Errors: {len(errors)}")
    
    if needs_reencoding:
        print()
        print("="*70)
        print("VIDEOS THAT NEED RE-ENCODING")
        print("="*70)
        for session_id, video_path, codec in needs_reencoding:
            print(f"  • {session_id} ({codec})")
        
        print()
        print("To re-encode these videos, run:")
        print()
        for session_id, video_path, codec in needs_reencoding[:5]:  # Show first 5
            print(f'  python reencode_video_h264.py "{video_path}"')
        
        if len(needs_reencoding) > 5:
            print(f"  ... and {len(needs_reencoding) - 5} more")
        
        print()
        print("Or re-encode all at once:")
        print(f"  python batch_reencode_all.py")


if __name__ == '__main__':
    main()

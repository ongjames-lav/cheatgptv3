#!/usr/bin/env python3
"""Diagnose video playback issues - check codec, format, and browser compatibility"""

import cv2
import os
from pathlib import Path

session_id = "single_1761069614"
video_path = Path("web_app/results") / session_id / f"processed_{session_id}.mp4"

print("="*70)
print("VIDEO DIAGNOSTICS")
print("="*70)

if not video_path.exists():
    print(f"❌ Video file not found: {video_path}")
    exit(1)

print(f"📹 File: {video_path}")
print(f"📦 Size: {video_path.stat().st_size / (1024*1024):.2f} MB")

# Open video with OpenCV
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("❌ Cannot open video with OpenCV")
    exit(1)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
duration = frame_count / fps if fps > 0 else 0

# Decode fourcc
fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

print(f"\n📊 VIDEO PROPERTIES:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Frames: {frame_count}")
print(f"  Duration: {duration:.2f}s")
print(f"  Codec (FourCC): {fourcc_str} ({hex(fourcc)})")

# Try to read first frame
ret, frame = cap.read()
if ret:
    print(f"  First frame: ✅ Readable ({frame.shape})")
else:
    print(f"  First frame: ❌ Cannot read")

cap.release()

# Check codec compatibility
print(f"\n🔍 BROWSER COMPATIBILITY:")
print(f"  Codec: {fourcc_str}")

browser_compatible_codecs = ['avc1', 'h264', 'H264', 'X264', 'x264']
if fourcc_str in browser_compatible_codecs:
    print(f"  ✅ Codec is browser-compatible (H.264)")
else:
    print(f"  ⚠️  Codec may not be browser-compatible")
    print(f"     Expected: H.264/AVC1")
    print(f"     Got: {fourcc_str}")
    print(f"     🔧 Video may need re-encoding")

# Check if it's a valid MP4 container
print(f"\n📦 CONTAINER FORMAT:")
with open(video_path, 'rb') as f:
    header = f.read(100)
    if b'ftyp' in header:
        print(f"  ✅ Valid MP4 container (ftyp box found)")
        # Check MP4 brand
        ftyp_idx = header.find(b'ftyp')
        if ftyp_idx >= 0:
            brand = header[ftyp_idx+4:ftyp_idx+8].decode('ascii', errors='ignore')
            print(f"  Brand: {brand}")
    else:
        print(f"  ❌ Invalid MP4 container (no ftyp box)")

# Additional checks
print(f"\n🔧 RECOMMENDATIONS:")
print(f"  1. Check browser console (F12) for specific errors")
print(f"  2. Try different browser (Chrome, Firefox, Edge)")
print(f"  3. Check if video plays in VLC or similar player")

if fourcc_str not in browser_compatible_codecs:
    print(f"  4. ⚠️  RE-ENCODE VIDEO with H.264 codec:")
    print(f"     ffmpeg -i {video_path} -c:v libx264 -c:a aac -movflags +faststart output.mp4")

print("\n" + "="*70)

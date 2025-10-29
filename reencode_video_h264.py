#!/usr/bin/env python3
"""
Re-encode video with browser-compatible H.264 codec
This fixes videos that were encoded with FMP4 codec
"""

import cv2
import os
from pathlib import Path
import shutil

def reencode_video_h264(input_path, output_path=None, backup=True):
    """Re-encode video with H.264 codec for browser compatibility"""
    
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"temp_{input_path.name}"
    else:
        output_path = Path(output_path)
    
    print(f"🎬 Re-encoding: {input_path.name}")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Cannot open input video")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Properties: {width}x{height} @ {fps:.1f} FPS, {frame_count} frames")
    
    # Try H.264 codecs
    writer = None
    codec_options = [
        ('avc1', 'H.264 AVC1'),
        ('H264', 'H.264'),
        ('X264', 'X264')
    ]
    
    for codec_name, codec_desc in codec_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            test_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if test_writer.isOpened():
                writer = test_writer
                print(f"   ✅ Using {codec_desc} codec")
                break
            else:
                test_writer.release()
        except Exception as e:
            continue
    
    if writer is None or not writer.isOpened():
        print(f"   ❌ Cannot create H.264 video writer")
        cap.release()
        return False
    
    # Re-encode frame by frame
    print(f"   🔄 Re-encoding frames...")
    processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        writer.write(frame)
        processed += 1
        
        if processed % 100 == 0:
            progress = (processed / frame_count) * 100
            print(f"   Progress: {processed}/{frame_count} ({progress:.1f}%)", end='\r')
    
    print(f"   Progress: {processed}/{frame_count} (100.0%)  ")
    
    cap.release()
    writer.release()
    
    # Verify output file
    test_cap = cv2.VideoCapture(str(output_path))
    if not test_cap.isOpened():
        print(f"   ❌ Output video cannot be opened")
        return False
    
    out_fourcc = int(test_cap.get(cv2.CAP_PROP_FOURCC))
    out_fourcc_str = "".join([chr((out_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    test_cap.release()
    
    print(f"   ✅ Output codec: {out_fourcc_str}")
    
    # Backup original and replace
    if backup:
        backup_path = input_path.parent / f"backup_{input_path.name}"
        print(f"   💾 Backing up original to: {backup_path.name}")
        shutil.copy2(input_path, backup_path)
    
    print(f"   🔄 Replacing original file...")
    shutil.move(str(output_path), str(input_path))
    
    print(f"   ✅ Done!")
    return True


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python reencode_video_h264.py <video_path>")
        print("\nExample:")
        print("  python reencode_video_h264.py web_app/results/single_1761069614/processed_single_1761069614.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print("="*70)
    print("VIDEO RE-ENCODER - H.264 for Browser Compatibility")
    print("="*70)
    
    if not Path(video_path).exists():
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    
    success = reencode_video_h264(video_path, backup=True)
    
    if success:
        print("\n✅ SUCCESS! Video is now browser-compatible")
        print("   Refresh your browser and try playing the video again")
    else:
        print("\n❌ FAILED to re-encode video")
        sys.exit(1)


if __name__ == '__main__':
    main()

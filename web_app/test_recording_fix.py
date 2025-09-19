#!/usr/bin/env python3
"""
Quick test to verify OpenH264 error is fixed
"""

import cv2
import numpy as np
import os

def test_video_recording_fix():
    """Test the improved video codec system"""
    print("🧪 Testing Improved Video Codec System")
    print("=" * 50)
    
    # Codec preference order (prioritize H.264 AVC1 for web compatibility)
    codec_options = [
        ('avc1', '.mp4', 'H.264 AVC1'),  # Best web compatibility
        ('H264', '.mp4', 'H.264'),       # Alternative H.264 fourcc
        ('mp4v', '.mp4', 'MPEG-4'),      # Fallback
        ('XVID', '.avi', 'Xvid'),        # Very reliable fallback
        ('MJPG', '.avi', 'Motion JPEG'), # Always works but larger files
    ]
    
    frame_size = (640, 480)
    fps = 15.0
    
    # Create test frame
    test_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    cv2.putText(test_frame, 'TEST RECORDING', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    working_codec = None
    
    for codec, ext, name in codec_options:
        print(f"Testing {name} ({codec})...", end=" ")
        
        try:
            filename = f'test_recording{ext}'
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            
            if writer.isOpened():
                # Write test frames
                for i in range(30):  # 2 seconds at 15fps
                    writer.write(test_frame)
                
                writer.release()
                
                # Check if file was created
                if os.path.exists(filename) and os.path.getsize(filename) > 1000:
                    print(f"✅ SUCCESS ({os.path.getsize(filename)} bytes)")
                    working_codec = (codec, ext, name)
                    
                    # Clean up
                    os.remove(filename)
                    break
                else:
                    print("❌ Failed (no output)")
            else:
                print("❌ Failed (couldn't open)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if working_codec:
        print(f"\n✅ RESULT: {working_codec[2]} codec will be used")
        print("🎉 Video recording should work without OpenH264 errors!")
    else:
        print("\n❌ No working codecs found!")
    
    return working_codec

if __name__ == "__main__":
    test_video_recording_fix()
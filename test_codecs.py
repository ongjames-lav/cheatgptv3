#!/usr/bin/env python3
"""
Test OpenCV video codecs including OpenH264
"""

import cv2
import numpy as np
import os
import sys

def test_video_codecs():
    """Test which video codecs are available"""
    print("🧪 Testing Video Codecs")
    print("=" * 50)
    
    # Print OpenCV build info
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Python: {sys.version}")
    print()
    
    test_codecs = [
        ('H264', '.mp4', 'H.264 (OpenH264)'),
        ('mp4v', '.mp4', 'MPEG-4'),
        ('XVID', '.avi', 'Xvid'), 
        ('MJPG', '.avi', 'Motion JPEG'),
        ('DIVX', '.avi', 'DivX'),
    ]
    
    working_codecs = []
    frame_size = (640, 480)
    fps = 30.0
    
    # Create a test frame
    test_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    cv2.putText(test_frame, 'TEST FRAME', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    for codec, ext, name in test_codecs:
        print(f"Testing {name} ({codec})...", end=" ")
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            filename = f'test_{codec}{ext}'
            
            writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            
            if writer.isOpened():
                # Try to write a few frames
                for i in range(10):
                    writer.write(test_frame)
                
                writer.release()
                
                # Check if file was created and has content
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    print("✅ WORKING")
                    working_codecs.append((codec, ext, name))
                else:
                    print("❌ FAILED (no output)")
                
                # Clean up test file
                try:
                    os.remove(filename)
                except:
                    pass
            else:
                print("❌ FAILED (couldn't open)")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print()
    print("📊 RESULTS:")
    print("-" * 30)
    if working_codecs:
        print("✅ Working codecs:")
        for codec, ext, name in working_codecs:
            print(f"  • {name} ({codec})")
    else:
        print("❌ No working codecs found!")
    
    return working_codecs

def test_openh264_specifically():
    """Test OpenH264 codec specifically"""
    print("\n🎯 Testing OpenH264 Specifically")
    print("=" * 50)
    
    try:
        # Check if we can create the fourcc
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        print("✅ H264 fourcc created successfully")
        
        # Try to create a writer
        writer = cv2.VideoWriter('test_h264.mp4', fourcc, 30.0, (640, 480))
        
        if writer.isOpened():
            print("✅ OpenH264 VideoWriter opened successfully")
            
            # Create and write test frames
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            for i in range(30):  # 1 second of video
                writer.write(frame)
            
            writer.release()
            
            # Check file
            if os.path.exists('test_h264.mp4') and os.path.getsize('test_h264.mp4') > 0:
                file_size = os.path.getsize('test_h264.mp4')
                print(f"✅ OpenH264 test successful! File size: {file_size} bytes")
                
                # Clean up
                os.remove('test_h264.mp4')
                return True
            else:
                print("❌ OpenH264 failed: No output file created")
                
        else:
            print("❌ OpenH264 failed: Could not open VideoWriter")
            
    except Exception as e:
        print(f"❌ OpenH264 error: {e}")
    
    return False

if __name__ == "__main__":
    # Test all codecs
    working_codecs = test_video_codecs()
    
    # Test OpenH264 specifically
    h264_works = test_openh264_specifically()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Total working codecs: {len(working_codecs)}")
    print(f"   OpenH264 status: {'✅ Working' if h264_works else '❌ Not working'}")
    
    if not h264_works and working_codecs:
        recommended = working_codecs[0]
        print(f"   Recommended fallback: {recommended[2]} ({recommended[0]})")
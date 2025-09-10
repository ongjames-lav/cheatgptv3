"""Test adaptive video recording to fix fast forward issue.

This test demonstrates the improved video recording with adaptive FPS
that matches the actual frame processing rate.
"""

import cv2
import time
import logging
import numpy as np
from datetime import datetime
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheatgpt.adaptive_video_recorder import AdaptiveVideoRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_adaptive_video_recording():
    """Test adaptive video recording with simulated frames."""
    
    print("🎥 Testing Adaptive Video Recording (Anti-Fast-Forward)")
    print("=" * 60)
    
    # Initialize recorder
    recorder = AdaptiveVideoRecorder(videos_dir="videos", target_fps=15.0)
    
    # Simulate webcam capture
    frame_width, frame_height = 640, 480
    session_id = f"adaptive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Start recording
        success, video_path = recorder.start_recording(
            session_id=session_id,
            frame_size=(frame_width, frame_height),
            cam_id="test_cam"
        )
        
        if not success:
            print("❌ Failed to start recording")
            return
        
        print(f"✅ Recording started: {video_path}")
        
        # Simulate frame processing with varying delays (realistic scenario)
        frame_delays = [
            0.05,   # Fast processing (20 fps)
            0.1,    # Medium processing (10 fps)
            0.15,   # Slower processing (6.7 fps)
            0.08,   # Variable processing (12.5 fps)
            0.12,   # Variable processing (8.3 fps)
            0.06,   # Back to faster (16.7 fps)
        ]
        
        total_frames = 30
        
        for i in range(total_frames):
            # Create a test frame with overlay information
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)  # Dark gray background
            
            # Add frame info overlay
            frame_text = f"Frame {i+1}/{total_frames}"
            time_text = f"Time: {time.time():.2f}s"
            fps_text = f"Target FPS: {recorder.current_fps}"
            
            cv2.putText(frame, frame_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, time_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, fps_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add detection boxes (simulated)
            cv2.rectangle(frame, (200, 250), (400, 350), (0, 0, 255), 2)
            cv2.putText(frame, "Person (Suspicious)", (205, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Write frame
            success = recorder.write_frame(frame)
            if success:
                print(f"📹 Frame {i+1} recorded (FPS: {recorder.current_fps})")
            else:
                print(f"❌ Failed to record frame {i+1}")
            
            # Simulate realistic processing delay
            delay = frame_delays[i % len(frame_delays)]
            time.sleep(delay)
        
        # Stop recording
        success, session_info = recorder.stop_recording()
        
        if success:
            print("\n✅ Recording completed successfully!")
            print(f"📊 Session Statistics:")
            print(f"   Session ID: {session_info['session_id']}")
            print(f"   Video Path: {session_info['video_path']}")
            print(f"   Duration: {session_info['duration_seconds']:.2f}s")
            print(f"   Frames: {session_info['frame_count']}")
            print(f"   Video FPS: {session_info['video_fps']}")
            print(f"   Processing FPS: {session_info['actual_processing_fps']:.2f}")
            print(f"   Sync Efficiency: {session_info['fps_efficiency']:.1f}%")
            
            # Check file size
            if os.path.exists(session_info['video_path']):
                file_size = os.path.getsize(session_info['video_path']) / (1024 * 1024)
                print(f"   File Size: {file_size:.2f} MB")
                print(f"\n🎬 Video saved: {session_info['video_path']}")
                print("   This video should play at normal speed (not fast forward)!")
            else:
                print("❌ Video file not found")
        else:
            print("❌ Failed to stop recording")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
    
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    test_adaptive_video_recording()

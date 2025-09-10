"""Test script for CheatGPT video recording functionality.

This script demonstrates:
- Session management with video recording
- Real-time webcam processing with overlays
- Database storage of session metadata and hotspots
- Video output with bounding boxes and labels
"""

import os
import sys
import time
import cv2
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_video_recording():
    """Test video recording functionality."""
    logger.info("🎥 Testing CheatGPT Video Recording Functionality")
    logger.info("=" * 60)
    
    try:
        from cheatgpt.engine import Engine
        
        # Initialize engine
        engine = Engine()
        logger.info("✅ Engine initialized successfully")
        
        # Check camera availability
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ No webcam detected")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"📷 Camera detected: {width}x{height}")
        
        # Start session with video recording
        session_id = engine.start_session(
            cam_id="test_recording",
            frame_size=(width, height)
        )
        
        if not session_id:
            logger.error("❌ Failed to start recording session")
            cap.release()
            return
        
        logger.info(f"🎬 Recording session started: {session_id}")
        logger.info("📹 Processing frames with video recording...")
        logger.info("   Press 'q' to stop recording")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to capture frame")
                    break
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame through engine (includes video recording)
                overlay_frame, events = engine.process_frame(
                    frame=frame,
                    cam_id="test_recording",
                    ts=time.time()
                )
                
                frame_count += 1
                
                # Add recording status overlay
                recording_status = engine.get_session_status()
                if recording_status.get('recording', {}).get('recording', False):
                    cv2.putText(overlay_frame, "🔴 REC", (width - 100, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add frame counter
                cv2.putText(overlay_frame, f"Frame: {frame_count}", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display
                cv2.imshow('CheatGPT Video Recording Test', overlay_frame)
                
                # Check for events
                if events:
                    critical_events = [e for e in events if e.get('severity') == 'red']
                    if critical_events:
                        logger.warning(f"🚨 Critical events detected: {len(critical_events)}")
                
                # Exit on 'q' key or after 30 seconds
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or (time.time() - start_time) > 30:
                    break
        
        except KeyboardInterrupt:
            logger.info("🛑 Recording interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Stop session and get results
            session_info = engine.stop_session()
            
            logger.info("📊 Recording Session Results:")
            logger.info(f"   Session ID: {session_id}")
            logger.info(f"   Duration: {session_info.get('duration', 0):.1f} seconds")
            logger.info(f"   Frames recorded: {session_info.get('frame_count', frame_count)}")
            logger.info(f"   Video file: {session_info.get('video_path', 'Not recorded')}")
            logger.info(f"   Video size: {get_file_size(session_info.get('video_path', ''))} MB")
            logger.info(f"   Hotspots detected: {session_info.get('hotspot_count', 0)}")
            
            # Verify video file
            video_path = session_info.get('video_path')
            if video_path and os.path.exists(video_path):
                logger.info(f"✅ Video file created successfully: {video_path}")
                
                # Test video playback
                test_cap = cv2.VideoCapture(video_path)
                if test_cap.isOpened():
                    frame_count_recorded = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps_recorded = test_cap.get(cv2.CAP_PROP_FPS)
                    logger.info(f"📹 Recorded video: {frame_count_recorded} frames @ {fps_recorded:.1f} fps")
                    test_cap.release()
                else:
                    logger.warning("⚠️ Could not open recorded video for verification")
            else:
                logger.warning("⚠️ Video file not found or not created")
    
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("   Please ensure CheatGPT modules are properly installed")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def get_file_size(filepath: str) -> float:
    """Get file size in MB."""
    try:
        if os.path.exists(filepath):
            size_bytes = os.path.getsize(filepath)
            return size_bytes / (1024 * 1024)  # Convert to MB
        return 0.0
    except Exception:
        return 0.0

def test_batch_video_processing():
    """Test batch video processing with recording."""
    logger.info("🎬 Testing Batch Video Processing")
    
    # Create a test video if none exists
    test_video_path = "test_input_video.mp4"
    if not os.path.exists(test_video_path):
        logger.info("📹 Creating test video...")
        create_test_video(test_video_path)
    
    if not os.path.exists(test_video_path):
        logger.warning("⚠️ No test video available for batch processing")
        return
    
    try:
        from cheatgpt.app import CheatGPTApp
        
        app = CheatGPTApp()
        app.demo_batch_processing(test_video_path)
        
    except Exception as e:
        logger.error(f"❌ Batch processing test failed: {e}")

def create_test_video(output_path: str, duration_seconds: int = 10):
    """Create a simple test video for batch processing."""
    try:
        # Try to use webcam to create test video
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("❌ Cannot create test video - no webcam available")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_to_record = int(fps * duration_seconds)
        logger.info(f"🎥 Recording {frames_to_record} frames for test video...")
        
        for i in range(frames_to_record):
            ret, frame = cap.read()
            if ret:
                # Add frame number overlay
                cv2.putText(frame, f"Test Frame {i+1}/{frames_to_record}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
        
        cap.release()
        out.release()
        
        logger.info(f"✅ Test video created: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create test video: {e}")

def main():
    """Main test function."""
    print("🎓 CheatGPT Video Recording Test Suite")
    print("=" * 50)
    
    # Test 1: Real-time video recording
    print("\n1️⃣ Testing real-time video recording...")
    test_video_recording()
    
    # Test 2: Batch video processing (optional)
    print("\n2️⃣ Testing batch video processing...")
    # test_batch_video_processing()  # Uncomment to test batch processing
    
    print("\n✅ Video recording tests completed!")

if __name__ == "__main__":
    main()

"""Debug script to analyze phone detection in a specific video."""
import cv2
import logging
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

from cheatgpt.engines.engine_hybrid import EngineHybrid

def debug_video_phone_detection(video_path: str, num_frames: int = 300):
    """Debug phone detection in a video by processing first N frames."""
    print(f"\n{'='*80}")
    print(f"DEBUG PHONE DETECTION ANALYSIS")
    print(f"{'='*80}\n")
    print(f"📹 Video: {video_path}")
    print(f"🎯 Analyzing first {num_frames} frames for phone detection\n")
    
    # Initialize engine
    print("🚀 Initializing detection engine...")
    engine = EngineHybrid()
    print("✅ Engine initialized\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video: {video_path}")
        return
    AttributeError
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📊 Video info: {total_frames} frames @ {fps:.2f} FPS\n")
    
    frame_count = 0
    phone_detection_frames = []
    phone_event_frames = []
    
    print("🔍 Starting frame-by-frame analysis...\n")
    print(f"{'Frame':<8} {'YOLO Phones':<15} {'Events':<40}")
    print(f"{'-'*8} {'-'*15} {'-'*40}")
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        _, events = engine.process_frame(frame, ts=frame_count / fps)
        
        # Track phone detection frames
        # Note: We'll rely on debug logs to see YOLO detections
        
        # Track phone events
        phone_events = [e for e in events if 'phone' in e.get('event_type', '').lower()]
        if phone_events:
            phone_event_frames.append(frame_count)
            event_str = ', '.join([e['event_type'] for e in phone_events])
            print(f"{frame_count:<8} {'CHECK LOGS':<15} {event_str:<40}")
    
    cap.release()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"📊 Frames analyzed: {frame_count}/{total_frames}")
    print(f"📱 Phone events detected: {len(phone_event_frames)} frames")
    
    if phone_event_frames:
        print(f"\n✅ Phone events at frames: {phone_event_frames[:10]}{'...' if len(phone_event_frames) > 10 else ''}")
    else:
        print(f"\n❌ NO PHONE EVENTS DETECTED")
        print(f"\n⚠️  Check the logs above for:")
        print(f"   1. '📱 YOLO DETECTED' - Are phones detected by YOLO?")
        print(f"   2. '📱 CALLING PHONE PROXIMITY CHECK' - Are phones passed to pose detector?")
        print(f"   3. '🔍 PHONE PROXIMITY CHECK' - Is proximity calculation working?")
        print(f"   4. '✅ PHONE MATCHED TO PERSON' - Are phones matching persons?")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    # Path to the uploaded video
    video_path = r"C:\Users\admin\Downloads\179948d7-affa-4976-92b9-df39ea79a7a4.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print(f"   Please provide the correct path to your uploaded video.")
        sys.exit(1)
    
    # Analyze first 300 frames (about 10 seconds at 30fps)
    debug_video_phone_detection(video_path, num_frames=300)

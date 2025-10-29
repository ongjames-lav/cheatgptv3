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
    all_events = []
    
    print("🔍 Starting frame-by-frame analysis...\n")
    print(f"{'Frame':<8} {'Phone Events':<60}")
    print(f"{'-'*8} {'-'*60}")
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        _, events = engine.process_frame(frame, ts=frame_count / fps)
        
        # Track ALL events
        all_events.extend(events)
        
        # Track phone events
        phone_events = [e for e in events if 'phone' in e.get('event_type', '').lower()]
        if phone_events:
            phone_event_frames.append(frame_count)
            for event in phone_events:
                event_str = f"{event.get('event_type', 'Unknown')} - {event.get('person_id', 'Unknown')}"
                print(f"{frame_count:<8} {event_str:<60}")
    
    cap.release()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"📊 Frames analyzed: {frame_count}/{total_frames}")
    print(f"� Total events detected: {len(all_events)}")
    print(f"�📱 Phone events detected: {len(phone_event_frames)} frames")
    
    # Count events by type
    event_types = {}
    for event in all_events:
        event_type = event.get('event_type', 'Unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    if event_types:
        print(f"\n📊 Event breakdown:")
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {event_type}: {count}")
    
    if phone_event_frames:
        print(f"\n✅ Phone events at frames: {phone_event_frames[:20]}{'...' if len(phone_event_frames) > 20 else ''}")
    else:
        print(f"\n⚠️ NO PHONE USAGE EVENTS DETECTED")
        print(f"\nPossible reasons:")
        print(f"   1. Phones detected but not meeting temporal requirements (need 8 consecutive frames)")
        print(f"   2. Phones matched to persons but failing phone_flag criteria")
        print(f"   3. Check logs for '✅ PHONE MATCHED TO PERSON' messages")
    
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

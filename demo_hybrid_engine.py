"""Simple demo showing the Research-Based Hybrid Engine in action.

This demonstrates the key improvements:
- 30 FPS smooth live stream
- 10 FPS detection with research-based rules
- Supervision ByteTrack for robust multi-person tracking
- No LSTM dependencies
- Real classroom behavior analysis
"""

import cv2
import time
import numpy as np
import sys
import os
import logging

# Setup logging to see ByteTrack activity
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def run_demo():
    """Run a simple demo with webcam or synthetic data."""
    
    print("🚀 Starting CheatGPT Research-Based Engine Demo")
    print("=" * 70)
    print("🔍 USING SUPERVISION BYTETRACK FOR MULTI-PERSON TRACKING")
    print("=" * 70)
    
    # Import engine (same pattern as test file)
    from cheatgpt.engines.engine_hybrid import EngineHybrid
    
    # Initialize engine
    print("\n⚙️  Initializing engine with ByteTrack...")
    engine = EngineHybrid()
    print("✅ Engine initialized - ByteTrack ready for tracking!\n")
    
    # Try to use webcam, fallback to synthetic data
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    
    if use_webcam:
        print("📹 Using webcam input")
        # Set camera properties for consistent framerate
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start recording session
        session_id = engine.start_session("webcam_demo", (640, 480))
        print(f"📝 Started session: {session_id}")
    else:
        print("📊 Using synthetic data (no webcam detected)")
        cap.release()
    
    frame_count = 0
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    
    try:
        print("\n🎬 Demo running... Press 'q' to quit")
        print("\n" + "=" * 70)
        print("WATCH THE CONSOLE FOR BYTETRACK ACTIVITY:")
        print("  🔄 BYTETRACK INPUT: Shows persons sent to ByteTrack")
        print("  ✅ BYTETRACK OUTPUT: Shows tracked persons with persistent IDs")
        print("  🎯 POSE ANALYSIS: Shows ByteTrack IDs flowing through pipeline")
        print("=" * 70)
        print("\nWATCH THE VIDEO FOR DETECTION:")
        print("  - Green boxes: Normal behavior")
        print("  - Yellow/Orange boxes: Suspicious behavior") 
        print("  - Red boxes: Cheating detected")
        print()
        
        # Track person IDs to verify ByteTrack persistence
        seen_track_ids = set()
        max_simultaneous_tracks = 0
        
        while True:
            if use_webcam:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
            else:
                # Create synthetic frame
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Add some synthetic "person" rectangles for demo
                cv2.rectangle(frame, (200, 150), (400, 350), (100, 150, 200), -1)
                cv2.putText(frame, "Synthetic Person", (210, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Process frame with hybrid engine (ByteTrack runs here!)
            overlay_frame, events = engine.process_frame(frame)
            
            # Track ByteTrack statistics
            if hasattr(engine, 'last_tracks') and engine.last_tracks:
                current_track_ids = {t.get('track_id') for t in engine.last_tracks}
                seen_track_ids.update(current_track_ids)
                max_simultaneous_tracks = max(max_simultaneous_tracks, len(current_track_ids))
            
            # Handle events
            for event in events:
                severity_emoji = {
                    'red': '🚨',
                    'orange': '⚠️', 
                    'yellow': '💛'
                }.get(event['severity'], '📊')
                
                print(f"{severity_emoji} {event['event_type']}: {event['details']}")
            
            # Add demo info overlay
            info_y = 60
            cv2.putText(overlay_frame, f"CheatGPT Research-Based Engine Demo", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # FPS counter
            fps_counter += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
                
                cv2.putText(overlay_frame, f"FPS: {fps:.1f}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Detection status
            is_detection_frame = (engine.frame_count % engine.skip_rate == 0)
            detection_text = "DETECTING" if is_detection_frame else "TRACKING"
            detection_color = (0, 255, 255) if is_detection_frame else (255, 255, 255)
            cv2.putText(overlay_frame, detection_text, 
                       (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
            
            # Show frame
            cv2.imshow('CheatGPT Research-Based Engine Demo', overlay_frame)
            
            frame_count += 1
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Auto-exit for synthetic demo after 300 frames (10 seconds)
            if not use_webcam and frame_count >= 300:
                print("Synthetic demo completed")
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Cleanup
        if use_webcam:
            cap.release()
        cv2.destroyAllWindows()
        
        # Stop session and get stats
        if use_webcam:
            session_info = engine.stop_session()
            print(f"\n📊 Session ended: {session_info}")
        
        # Final statistics
        total_time = time.time() - start_time
        stats = engine.get_statistics()
        
        print(f"\n📈 Demo Statistics:")
        print(f"   Total frames: {frame_count}")
        print(f"   Duration: {total_time:.1f}s")
        print(f"   Average FPS: {frame_count/total_time:.1f}")
        print(f"   Engine performance: {stats['performance']['avg_fps']:.1f} FPS")
        print(f"   Detection latency: {stats['performance']['avg_detection_time_ms']:.1f}ms")
        print(f"   Active persons: {stats['rule_engine']['active_persons']}")
        
        # ByteTrack statistics
        print(f"\n🔍 ByteTrack Statistics:")
        print(f"   Total unique track IDs seen: {len(seen_track_ids)}")
        print(f"   Max simultaneous tracks: {max_simultaneous_tracks}")
        print(f"   Track IDs used: {sorted(seen_track_ids) if seen_track_ids else 'None'}")
        
        if seen_track_ids:
            print(f"\n✅ BYTETRACK WORKING: Persistent IDs were assigned!")
        else:
            print(f"\n⚠️  No persons detected in this session")

def main():
    """Main demo function."""
    try:
        run_demo()
        print("\n✅ Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
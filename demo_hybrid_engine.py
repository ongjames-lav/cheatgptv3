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

# Setup logging to see ByteTrack activity (WARNING level for better FPS)
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING for better performance
    format='%(levelname)s - %(name)s - %(message)s'
)

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def create_clean_ui(frame, fps, person_count, minimized):
    """Create a clean, professional UI overlay for the demo - optimized for performance."""
    h, w = frame.shape[:2]
    
    if minimized:
        # Minimal HUD - just FPS in corner
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        cv2.putText(frame, "Press 'M' to expand", (w - 180, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        return frame
    
    # Clean mode - NO HEADER, just minimal controls hint
    # Minimal hint in bottom right
    hint_text = "M:minimize | F:fullscreen | Q:quit"
    cv2.putText(frame, hint_text, (w - 310, h - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    
    return frame

def run_demo():
    """Run a simple demo with webcam or synthetic data."""
    
    print("🚀 CheatGPT Demo - Press 'Q' to quit, 'M' to minimize, 'F' for fullscreen")
    
    # Import engine (same pattern as test file)
    from cheatgpt.engines.engine_hybrid import EngineHybrid
    
    # Initialize engine
    engine = EngineHybrid()
    
    # Try to use webcam, fallback to synthetic data
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    
    if use_webcam:
        # Set camera properties for higher resolution
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # DEMO MODE: No recording for better FPS (disk I/O disabled)
        session_id = None
    else:
        cap.release()
        session_id = None
    
    frame_count = 0
    start_time = time.time()
    fps_counter = 0
    last_fps_time = start_time
    current_fps = 0.0
    
    # Create fullscreen window
    window_name = 'CheatGPT Classroom Monitoring System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Track person IDs to verify ByteTrack persistence
    seen_track_ids = set()
    max_simultaneous_tracks = 0
    
    # UI state
    minimized = False  # Start with full HUD
    is_fullscreen = True  # Start in fullscreen mode
    
    try:
        
        while True:
            if use_webcam:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
            else:
                # Create synthetic frame
                frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                
                # Add some synthetic "person" rectangles for demo
                cv2.rectangle(frame, (400, 200), (600, 500), (100, 150, 200), -1)
                cv2.putText(frame, "Synthetic Person", (420, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Process frame with hybrid engine (ByteTrack runs here!)
            overlay_frame, events = engine.process_frame(frame)
            
            # Track ByteTrack statistics
            if hasattr(engine, 'last_tracks') and engine.last_tracks:
                current_track_ids = {t.get('track_id') for t in engine.last_tracks}
                seen_track_ids.update(current_track_ids)
                max_simultaneous_tracks = max(max_simultaneous_tracks, len(current_track_ids))
            
            # Handle events (console logging only - NOT saving to database)
            if events:
                for event in events:
                    severity_emoji = {
                        'red': '🚨',
                        'orange': '⚠️', 
                        'yellow': '💛'
                    }.get(event['severity'], '📊')
                    
                    # Clean, concise event logging
                    timestamp = time.strftime('%H:%M:%S', time.localtime(event.get('timestamp', time.time())))
                    person_id = event['person_id'].replace('person_', 'P')
                    print(f"[{timestamp}] {severity_emoji} {person_id}: {event['event_type']} - {event['details']}")
            
            # Calculate FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                current_fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            
            # Create clean UI overlay (optimized, no event counter)
            overlay_frame = create_clean_ui(
                overlay_frame, 
                current_fps,
                len(engine.last_tracks) if hasattr(engine, 'last_tracks') else 0,
                minimized
            )
            
            # Show frame (optimized display)
            cv2.imshow(window_name, overlay_frame)
            
            frame_count += 1
            
            # Handle key presses with minimal wait for better FPS
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('m'):
                minimized = not minimized
            elif key == ord('f'):
                # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    # Set window to a reasonable size when not fullscreen
                    cv2.resizeWindow(window_name, 1280, 720)
            
            # Auto-exit for synthetic demo after 300 frames (10 seconds)
            if not use_webcam and frame_count >= 300:
                break
                
    except KeyboardInterrupt:
        pass
    
    finally:
        # Cleanup
        if use_webcam:
            cap.release()
        cv2.destroyAllWindows()
        
        # No session to stop in demo mode (recording disabled for FPS)
        
        # Final statistics
        total_time = time.time() - start_time
        stats = engine.get_statistics()
        
        print(f"\n📈 Demo Statistics:")
        print(f"   Total frames: {frame_count}")
        print(f"   Duration: {total_time:.1f}s")
        print(f"   Average FPS: {frame_count/total_time:.1f}")
        print(f"   Engine performance: {stats['performance']['avg_fps']:.1f} FPS")
        print(f"   Detection latency: {stats['performance']['avg_detection_time_ms']:.1f}ms")
        # Removed active persons count to keep console clean
        
        # ByteTrack statistics
        print(f"\n🔍 ByteTrack Statistics:")
        # Removed total track IDs count to keep console clean
        print(f"   Max simultaneous tracks: {max_simultaneous_tracks}")
        # Removed detailed track IDs list to keep console clean
        
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
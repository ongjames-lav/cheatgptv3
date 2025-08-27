#!/usr/bin/env python3

import cv2
import time
import logging
from cheatgpt.engine import Engine

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

def test_temporal_detection():
    """Test temporal detection with simple webcam input."""
    
    print("🚀 Starting Temporal Detection Debug Test")
    print("=" * 60)
    
    # Initialize CheatGPT engine
    engine = Engine()
    
    # Print temporal settings
    print(f"📊 Temporal Settings:")
    print(f"   Enabled: {engine.temporal_cheating_enabled}")
    print(f"   Threshold: {engine.temporal_cheating_threshold}s")
    print(f"   Cooldown: {engine.cheating_cooldown_duration}s")
    print("=" * 60)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Process frame
            try:
                results = engine.process_frame(frame, current_time)
                
                # Debug temporal analysis specifically
                if results is not None and isinstance(results, dict) and 'pose_analysis' in results:
                    poses = results['pose_analysis']
                    if poses:
                        for pose in poses:
                            person_id = pose.get('person_id', 0)
                            looking_flag = pose.get('looking_around_flag', False)
                            lean_flag = pose.get('optimized_lean_flag', False)
                            
                            if looking_flag or lean_flag:
                                print(f"🔍 Frame {frame_count}: Person {person_id} - Looking: {looking_flag}, Leaning: {lean_flag}")
                                
                                # Check if temporal analysis was called
                                if hasattr(engine, 'looking_around_history') and hasattr(engine, 'leaning_history'):
                                    looking_count = len(engine.looking_around_history.get(person_id, []))
                                    leaning_count = len(engine.leaning_history.get(person_id, []))
                                    print(f"   📊 History: Looking={looking_count}, Leaning={leaning_count}")
                                    
                                    # Calculate durations manually
                                    if person_id in engine.looking_around_history and len(engine.looking_around_history[person_id]) > 0:
                                        looking_duration = engine._calculate_behavior_duration(engine.looking_around_history[person_id], current_time)
                                        print(f"   ⏱️ Looking Duration: {looking_duration:.1f}s")
                                        
                                        if looking_duration > 10.0:
                                            print(f"   🚨 SHOULD ESCALATE! Duration: {looking_duration:.1f}s > Threshold: {engine.temporal_cheating_threshold * 1.5:.1f}s")
                
                # Check for temporal events
                if results is not None and isinstance(results, dict) and 'temporal_events' in results:
                    temporal_events = results['temporal_events']
                    if temporal_events:
                        print(f"🎯 TEMPORAL EVENT DETECTED: {temporal_events}")
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
                import traceback
                traceback.print_exc()
            
            # Show frame
            cv2.imshow('Temporal Debug Test', frame)
            
            # Break on 'q' key or after 30 seconds
            if cv2.waitKey(1) & 0xFF == ord('q') or elapsed > 30:
                break
                
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        runtime = time.time() - start_time
        print(f"\n📊 Debug Test Complete:")
        print(f"   Runtime: {runtime:.1f}s")
        print(f"   Frames: {frame_count}")
        print(f"   FPS: {frame_count/runtime:.1f}")

if __name__ == "__main__":
    test_temporal_detection()

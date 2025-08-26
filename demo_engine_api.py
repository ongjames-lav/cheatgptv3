"""Simple demo showing the CheatGPT3 Engine API usage."""
import os
import sys
import cv2
import time

# Add the cheatgpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cheatgpt'))

def demo_engine_api():
    """Demonstrate the Engine API as specified."""
    print("🎓 CheatGPT3 Engine API Demo")
    print("=" * 40)
    
    # Import and initialize
    from cheatgpt.engine import Engine
    
    print("1. Initializing Engine...")
    engine = Engine()
    print(f"   ✓ Engine ready: {engine}")
    
    # Load test frame
    print("\n2. Loading test frame...")
    frame = cv2.imread("original_test_image.jpg")
    print(f"   ✓ Frame loaded: {frame.shape}")
    
    # Demonstrate API usage
    print("\n3. API Usage Demo:")
    print("   API: process_frame(frame, cam_id, ts) -> overlay_frame, events")
    
    # Example 1: Basic usage
    print("\n   Example 1: Basic API call")
    cam_id = "classroom_cam_01"
    ts = time.time()
    
    overlay_frame, events = engine.process_frame(frame, cam_id, ts)
    
    print(f"   Input:")
    print(f"     frame: {frame.shape} image")
    print(f"     cam_id: '{cam_id}'")
    print(f"     ts: {ts}")
    print(f"   Output:")
    print(f"     overlay_frame: {overlay_frame.shape} image with annotations")
    print(f"     events: {len(events)} violation events")
    
    # Show events
    if events:
        print("   Events detected:")
        for i, event in enumerate(events):
            print(f"     Event {i+1}:")
            print(f"       Track: {event['track_id']}")
            print(f"       Type: {event['event_type']}")
            print(f"       Severity: {event['severity']}")
            print(f"       Camera: {event['cam_id']}")
            print(f"       Confidence: {event['confidence']:.0%}")
    
    # Example 2: Multiple frame processing
    print("\n   Example 2: Processing video sequence")
    
    for frame_num in range(1, 6):
        ts = time.time() + frame_num * 0.1  # 10 FPS simulation
        cam_id = f"exam_room_{frame_num % 2 + 1}"
        
        overlay_frame, events = engine.process_frame(frame, cam_id, ts)
        
        violation_types = [e['event_type'] for e in events]
        print(f"     Frame {frame_num}: {len(events)} events {violation_types}")
    
    # Example 3: Evidence saving demonstration
    print("\n   Example 3: Evidence saving (when cheating detected)")
    
    # Process enough frames to potentially trigger cheating
    cheating_detected = False
    for i in range(10):
        ts = time.time() + i * 0.1
        overlay_frame, events = engine.process_frame(frame, "evidence_demo", ts)
        
        # Check for cheating events
        cheating_events = [e for e in events if e['severity'] == 'Cheating']
        if cheating_events:
            cheating_detected = True
            print(f"     🚨 CHEATING DETECTED on frame {i+1}!")
            print(f"       Tracks involved: {[e['track_id'] for e in cheating_events]}")
            print(f"       Evidence automatically saved to: {engine.evidence_dir}")
            break
    
    if not cheating_detected:
        print("     No cheating detected in this sequence")
    
    # Show final visualization
    print("\n4. Saving Final Visualization...")
    final_overlay, final_events = engine.process_frame(frame, "demo_final", time.time())
    
    output_path = "engine_api_demo.jpg"
    cv2.imwrite(output_path, final_overlay)
    print(f"   ✓ Demo result saved to {output_path}")
    
    # Show statistics
    print("\n5. Engine Statistics:")
    stats = engine.get_statistics()
    print(f"   Frames processed: {stats['frame_count']}")
    print(f"   Active tracks: {stats['active_tracks']}")
    print(f"   Active violations: {stats['active_violations']}")
    print(f"   Average FPS: {stats['performance']['fps']:.1f}")
    
    print("\n✅ Engine API Demo Complete!")
    print(f"🎯 Key Features Demonstrated:")
    print(f"   • Full pipeline integration (YOLO → Tracker → Pose → Policy)")
    print(f"   • Bounding boxes with severity colors")
    print(f"   • Automatic evidence saving for cheating")
    print(f"   • Real-time processing capability")
    print(f"   • Comprehensive event reporting")

def show_api_reference():
    """Show API reference documentation."""
    print("\n📚 CheatGPT3 Engine API Reference")
    print("=" * 40)
    
    print("""
🔧 Engine Initialization:
   from cheatgpt.engine import Engine
   engine = Engine()

📹 Frame Processing:
   overlay_frame, events = engine.process_frame(frame, cam_id, ts)
   
   Parameters:
     frame (np.ndarray): Input video frame (BGR format)
     cam_id (str): Camera identifier (e.g., "classroom_01")
     ts (float, optional): Timestamp (uses current time if None)
   
   Returns:
     overlay_frame (np.ndarray): Frame with bounding boxes and labels
     events (List[Dict]): List of violation events
   
📊 Event Structure:
   {
     'timestamp': float,        # Event timestamp
     'cam_id': str,            # Camera identifier
     'track_id': str,          # Person track ID
     'event_type': str,        # Violation type
     'severity': str,          # Severity level
     'confidence': float,      # Detection confidence
     'bbox': List[float],      # Bounding box [x1,y1,x2,y2]
     'details': str            # Additional information
   }

🎨 Severity Colors:
   • Green: Normal behavior
   • Yellow: Minor violations (Leaning, Looking Around)
   • Orange: Phone Use
   • Red: Cheating

💾 Evidence Saving:
   Automatic when cheating detected:
   - Saves to uploads/evidence/
   - Filename: cheating_{cam_id}_{timestamp}_{track_ids}.jpg
   - Database entry created automatically

📈 Statistics:
   stats = engine.get_statistics()
   # Returns performance metrics, track counts, violation distribution
""")

if __name__ == "__main__":
    demo_engine_api()
    show_api_reference()

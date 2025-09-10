#!/usr/bin/env python3
"""
Demo Script for Hotspot Overlay System
Test the hotspot overlay functionality with simulated events
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from cheatgpt.overlays.hotspot_overlay import HotspotOverlay, EngineOverlayIntegration

def create_demo_frame(width=640, height=480, frame_number=0):
    """Create a demo frame with some visual elements"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background gradient
    for y in range(height):
        intensity = int(30 + (y / height) * 50)
        frame[y, :] = (intensity, intensity, intensity)
    
    # Add some visual elements
    cv2.putText(frame, f"Frame {frame_number}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.putText(frame, "Hotspot Overlay Demo", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw person rectangles
    cv2.rectangle(frame, (150, 120), (250, 350), (100, 100, 100), 2)
    cv2.putText(frame, "Person 1", (155, 140), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.rectangle(frame, (350, 100), (450, 320), (100, 100, 100), 2)
    cv2.putText(frame, "Person 2", (355, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def simulate_events(overlay: HotspotOverlay, timestamp: float, frame_number: int):
    """Simulate various events for demonstration"""
    events = []
    
    # Pattern 1: Looking around every 2 seconds
    if frame_number % 60 == 0:  # Every 2 seconds at 30fps
        events.append({
            'event_type': 'suspicious_looking',
            'person_id': 'person_001',
            'bbox': (150, 120, 100, 230),
            'confidence': 0.85,
            'additional_data': {'head_yaw': 15, 'direction': 'left'}
        })
    
    # Pattern 2: Gesture detection every 3 seconds
    if frame_number % 90 == 30:  # Every 3 seconds, offset by 1 second
        events.append({
            'event_type': 'suspicious_gesture',
            'person_id': 'person_001',
            'bbox': (150, 120, 100, 230),
            'confidence': 0.75,
            'additional_data': {'gesture_type': 'hand_to_face'}
        })
    
    # Pattern 3: Leaning detection every 4 seconds
    if frame_number % 120 == 60:  # Every 4 seconds, offset by 2 seconds
        events.append({
            'event_type': 'suspicious_lean',
            'person_id': 'person_002',
            'bbox': (350, 100, 100, 220),
            'confidence': 0.65,
            'additional_data': {'lean_angle': 8, 'direction': 'right'}
        })
    
    # Pattern 4: Phone detection every 5 seconds
    if frame_number % 150 == 90:  # Every 5 seconds, offset by 3 seconds
        events.append({
            'event_type': 'phone_detected',
            'person_id': 'person_002',
            'bbox': (350, 100, 100, 220),
            'confidence': 0.90,
            'additional_data': {'phone_type': 'smartphone'}
        })
    
    # Pattern 5: Temporal cheating (rare, high severity)
    if frame_number % 300 == 200:  # Every 10 seconds, offset by ~7 seconds
        events.append({
            'event_type': 'temporal_cheating',
            'person_id': 'person_001',
            'bbox': (150, 120, 100, 230),
            'confidence': 0.95,
            'additional_data': {
                'behaviors': ['looking', 'lean', 'gesture'],
                'duration': 8.5
            }
        })
    
    # Add events to overlay
    for event in events:
        overlay.add_event(
            event_type=event['event_type'],
            person_id=event['person_id'],
            bbox=event['bbox'],
            confidence=event['confidence'],
            additional_data=event['additional_data']
        )
    
    return events

def run_demo():
    """Run the hotspot overlay demonstration"""
    
    print("🎯 Hotspot Overlay Demo")
    print("=" * 50)
    print("This demo shows the hotspot overlay system in action.")
    print("\nFeatures demonstrated:")
    print("- Warning markers for suspicious events")
    print("- Color-coded event types")
    print("- Event timeline display")
    print("- Database storage")
    print("- Temporal event tracking")
    print("\nControls:")
    print("- ESC or 'q': Quit")
    print("- 's': Save timeline to file")
    print("- 'c': Clear all events")
    print("- SPACE: Pause/Resume")
    print("\nPress any key to start...")
    
    input()
    
    # Initialize overlay system
    overlay = HotspotOverlay(db_path="demo_events.db")
    
    # Demo settings
    fps = 30
    frame_interval = 1.0 / fps
    frame_number = 0
    paused = False
    
    start_time = time.time()
    
    print("\n🚀 Starting demo... Press ESC to quit")
    
    try:
        while True:
            current_time = time.time()
            timestamp = current_time - start_time
            
            if not paused:
                # Create demo frame
                frame = create_demo_frame(frame_number=frame_number)
                
                # Simulate events
                new_events = simulate_events(overlay, timestamp, frame_number)
                
                # Update overlay timestamp
                overlay.update_timestamp(timestamp)
                
                # Process frame with overlays
                processed_frame = overlay.process_frame(frame, timestamp, show_timeline=True)
                
                # Add demo info
                cv2.putText(processed_frame, f"Time: {timestamp:.1f}s", (20, height-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(processed_frame, f"Events this frame: {len(new_events)}", (20, height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                frame_number += 1
            else:
                # Use last processed frame when paused
                pass
            
            # Display frame
            cv2.imshow('Hotspot Overlay Demo', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(int(frame_interval * 1000)) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break
            elif key == ord('s'):
                # Save timeline
                timeline_file = f"demo_timeline_{int(timestamp)}.json"
                overlay.export_event_timeline(timeline_file)
                print(f"📄 Timeline saved to: {timeline_file}")
            elif key == ord('c'):
                # Clear events (reset overlay)
                overlay.active_events.clear()
                print("🧹 Events cleared")
            elif key == ord(' '):
                # Toggle pause
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
            
            # Maintain frame rate
            if not paused:
                time.sleep(max(0, frame_interval - (time.time() - current_time)))
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        
        # Export final timeline
        final_timeline_file = "demo_final_timeline.json"
        timeline_data = overlay.export_event_timeline(final_timeline_file)
        
        print("\n📊 Demo Complete!")
        print("=" * 50)
        print(f"Total Events: {timeline_data['session_info']['total_events']}")
        print(f"Duration: {timeline_data['session_info']['duration']:.1f}s")
        print(f"Timeline exported to: {final_timeline_file}")
        
        # Show event type summary
        if timeline_data['events']:
            event_types = {}
            for event in timeline_data['events']:
                event_type = event['event_type']
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            print("\nEvent Type Summary:")
            for event_type, count in event_types.items():
                print(f"  {event_type}: {count}")

if __name__ == "__main__":
    # Set frame height for calculations
    height = 480
    run_demo()

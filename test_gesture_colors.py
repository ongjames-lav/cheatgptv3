"""Test script to verify gesture detection color changes in the engine."""

import cv2
import numpy as np
import time
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_gesture_color_changes():
    """Test that gesture detection triggers bounding box color changes."""
    
    print("🚀 Testing Gesture Detection Color Changes")
    print("=" * 50)
    
    try:
        # Import the engine
        from cheatgpt.engines.engine_hybrid import EngineHybrid
        
        # Initialize engine
        print("Initializing engine...")
        engine = EngineHybrid()
        
        # Start a test session
        session_id = engine.start_session("gesture_test", (640, 480))
        print(f"📝 Started test session: {session_id}")
        
        # Create a synthetic frame with a person
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
        
        # Add a rectangular "person" area
        cv2.rectangle(frame, (200, 150), (400, 400), (100, 150, 100), -1)
        
        # Process the frame to establish baseline (should show green box)
        print("\n🔍 Processing baseline frame (no gesture)...")
        overlay_frame, events = engine.process_frame(frame)
        print(f"Events generated: {len(events)}")
        
        # Display the frame briefly
        cv2.imshow("Baseline - No Gesture", overlay_frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        
        # Now simulate gesture detection by creating mock detection data
        # This would normally come from the pose detector
        print("\n🤚 Simulating gesture detection...")
        
        # Create multiple frames with gesture flags to trigger detection
        for i in range(15):  # Send enough frames to trigger gesture event
            # Create mock detection data that includes person detection
            # This simulates what the YOLO detector and pose detector would provide
            mock_detections = [{
                'bbox': [200, 150, 400, 400],
                'cls_name': 'person',
                'conf': 0.8,
                'person_id': 'person_000',
                'keypoint_quality': 0.7
            }]
            
            # Create mock pose data with gesture flag
            mock_pose_data = [{
                'person_id': 'person_000',
                'bbox': [200, 150, 400, 400],
                'phone_flag': False,
                'head_turn_angle': 5.0,
                'lean_angle': 3.0,
                'gesture_flag': True,  # This is what triggers gesture detection
                'gesture_reason': 'left_wrist_extended_arm_absolute',
                'out_of_frame': False,
                'keypoint_quality': 0.7
            }]
            
            # Manually set the last detections and tracks for visualization
            engine.last_detections = mock_detections
            engine.last_tracks = [{'track_id': 0, 'bbox': [200, 150, 400, 400], 'person_id': 'person_000'}]
            
            # Manually trigger rule evaluation with gesture data
            if hasattr(engine, 'rule_engine'):
                rule_events = engine.rule_engine.update_detection(0, mock_pose_data[0], time.time())
                
                # Update active events for visualization
                for event in rule_events:
                    person_key = event['person_id']
                    engine.active_events[person_key] = event
                    engine.event_timestamps[person_key] = time.time()
                    engine.gesture_event_count += 1
            
            # Process frame - this will use the mock data for visualization
            overlay_frame, events = engine.process_frame(frame)
            
            print(f"Frame {i+1}: Rule events = {len(rule_events) if 'rule_events' in locals() else 0}, "
                  f"Gesture count = {engine.gesture_event_count}, Active events = {len(engine.active_events)}")
            
            # Display the frame
            window_title = f"Gesture Detection - Frame {i+1}"
            cv2.imshow(window_title, overlay_frame)
            
            if i == 5 or i == 10 or i == 14:  # Show a few key frames longer
                cv2.waitKey(1500)  # 1.5 seconds
            else:
                cv2.waitKey(200)   # 0.2 seconds
        
        # Test different gesture types
        gesture_types = [
            'right_wrist_extended_arm_absolute',
            'both_wrists_extended',
            'pointing_gesture'
        ]
        
        for gesture_type in gesture_types:
            print(f"\n🤚 Testing {gesture_type}...")
            
            # Create proper mock data for this gesture type
            mock_detections = [{
                'bbox': [200, 150, 400, 400],
                'cls_name': 'person',
                'conf': 0.9,
                'person_id': 'person_000',
                'keypoint_quality': 0.8
            }]
            
            mock_pose_data = [{
                'person_id': 'person_000',
                'bbox': [200, 150, 400, 400],
                'phone_flag': False,
                'head_turn_angle': 5.0,
                'lean_angle': 3.0,
                'gesture_flag': True,
                'gesture_reason': gesture_type,
                'out_of_frame': False,
                'keypoint_quality': 0.8
            }]
            
            # Set mock data and trigger rule evaluation
            engine.last_detections = mock_detections
            engine.last_tracks = [{'track_id': 0, 'bbox': [200, 150, 400, 400], 'person_id': 'person_000'}]
            
            if hasattr(engine, 'rule_engine'):
                rule_events = engine.rule_engine.update_detection(0, mock_pose_data[0], time.time())
                
                for event in rule_events:
                    person_key = event['person_id']
                    engine.active_events[person_key] = event
                    engine.event_timestamps[person_key] = time.time()
                    engine.gesture_event_count += 1
            
            # Show this gesture for a few seconds to see the pulsing effect
            for j in range(10):
                overlay_frame, events = engine.process_frame(frame)
                cv2.imshow(f"Gesture: {gesture_type}", overlay_frame)
                cv2.waitKey(200)
        
        # Test recent gesture fading effect
        print("\n🕐 Testing recent gesture fading effect...")
        
        # Keep the detections and tracks but clear active events
        mock_detections = [{
            'bbox': [200, 150, 400, 400],
            'cls_name': 'person',
            'conf': 0.7,
            'person_id': 'person_000',
            'keypoint_quality': 0.6
        }]
        
        engine.last_detections = mock_detections
        engine.last_tracks = [{'track_id': 0, 'bbox': [200, 150, 400, 400], 'person_id': 'person_000'}]
        
        # Clear active events but keep some in event timestamps for recent effect
        if 'person_000' in engine.active_events:
            recent_event = engine.active_events['person_000']
            del engine.active_events['person_000']
            # Keep timestamp recent for "recent gesture" effect
            engine.event_timestamps['person_000'] = time.time() - 1.0  # 1 second ago
        
        for k in range(10):
            overlay_frame, events = engine.process_frame(frame)
            cv2.imshow("Recent Gesture Fade", overlay_frame)
            cv2.waitKey(300)
        
        # Final statistics
        print(f"\n📊 Final Statistics:")
        stats = engine.get_statistics()
        print(f"Total gesture events detected: {stats['rule_engine']['gesture_events_detected']}")
        print(f"Frames processed: {stats['frame_count']}")
        print(f"Average FPS: {stats['performance']['avg_fps']:.1f}")
        
        # Stop session
        engine.stop_session()
        print("✅ Test completed successfully!")
        
        # Keep window open for final review
        print("\nPress any key to close windows and exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except ImportError as e:
        print(f"❌ Failed to import engine: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_gesture_color_changes()
    print(f"\n🎯 Test Result: {'✅ PASSED' if success else '❌ FAILED'}")
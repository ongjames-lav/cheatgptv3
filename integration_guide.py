"""Integration Guide for Research-Based Hybrid Engine

This guide shows how to integrate the new engine into your existing CheatGPT application.
"""

import cv2
import time
from cheatgpt.engines.engine_hybrid import EngineHybrid

def create_engine_instance():
    """Create and return the hybrid engine instance."""
    return EngineHybrid()

# 2. Example integration with web application
def integrate_with_web_app():
    """Example of how to integrate with the existing web app."""
    
    # In your main application file (e.g., cheatgpt_desktop.py)
    engine = EngineHybrid()
    
    # Start session for recording
    session_id = engine.start_session(
        cam_id="webcam_001",
        frame_size=(640, 480)  # Your camera resolution
    )
    
    # Main processing loop
    cap = cv2.VideoCapture(0)  # Webcam
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame (30 FPS live stream + 10 FPS detection)
            overlay_frame, events = engine.process_frame(frame)
            
            # Handle events in real-time
            for event in events:
                if event['severity'] == 'red':
                    print(f"🚨 CRITICAL: {event['event_type']} - {event['details']}")
                    # Send alert, log to database, etc.
                elif event['severity'] == 'orange':
                    print(f"⚠️ WARNING: {event['event_type']} - {event['details']}")
            
            # Display live stream
            cv2.imshow('CheatGPT Live Detection', overlay_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean shutdown
        cap.release()
        cv2.destroyAllWindows()
        session_info = engine.stop_session()
        print(f"Session completed: {session_info}")

# 3. Configuration options
ENGINE_CONFIG = {
    # Frame rate settings
    'target_fps': 30.0,
    'detection_fps': 10.0,  # Will be calculated from skip_rate
    'skip_rate': 3,
    
    # Detection thresholds
    'person_confidence': 0.4,
    'phone_confidence': 0.4,
    
    # Research-based behavior thresholds
    'phone_consecutive_frames': 3,      # ~0.5s at 10 FPS
    'head_turn_angle_threshold': 20.0,  # degrees
    'head_turn_frequency_threshold': 3, # times in window
    'head_turn_sustained_threshold': 2.0, # seconds
    'head_pitch_threshold': 25.0,       # degrees
    'head_pitch_frames_threshold': 12,  # ~1.2s at 10 FPS
    'hand_extended_frames_threshold': 15, # ~1.5s at 10 FPS
    'out_of_frame_threshold': 10,       # frames
    'normal_posture_reset_duration': 2.0, # seconds
    
    # Temporal analysis
    'window_size': 30,              # frames (~3s at 10 FPS)
    'confirmation_threshold': 2,    # confirmations needed
}

# 4. Event handling examples
def handle_cheating_event(event):
    """Handle different types of cheating events."""
    
    event_handlers = {
        'Phone Usage Detected': handle_phone_cheating,
        'Sustained Head Turning': handle_head_turn_cheating,
        'Frequent Head Turning': handle_frequent_looking,
        'Abnormal Looking Down': handle_abnormal_posture,
        'Suspicious Hand Activity': handle_hand_activity,
        'Hiding from Camera': handle_out_of_frame,
    }
    
    handler = event_handlers.get(event['event_type'])
    if handler:
        handler(event)
    else:
        print(f"Unknown event type: {event['event_type']}")

def handle_phone_cheating(event):
    """Handle phone usage detection."""
    print(f"📱 PHONE DETECTED: {event['details']}")
    # Immediate alert - highest priority
    # Could trigger:
    # - Instructor notification
    # - Screenshot capture
    # - Session flagging

def handle_head_turn_cheating(event):
    """Handle sustained head turning."""
    print(f"👀 SUSTAINED LOOKING: {event['details']}")
    # High priority - potential cheating
    # Could trigger:
    # - Warning notification
    # - Behavior pattern analysis

def handle_frequent_looking(event):
    """Handle frequent head movements."""
    print(f"🔄 FREQUENT LOOKING: {event['details']}")
    # Medium priority - monitor closely

def handle_abnormal_posture(event):
    """Handle abnormal looking down."""
    print(f"📝 ABNORMAL POSTURE: {event['details']}")
    # Medium priority - could be writing or cheating

def handle_hand_activity(event):
    """Handle suspicious hand activity."""
    print(f"🤚 HAND ACTIVITY: {event['details']}")
    # Medium priority - monitor for note passing

def handle_out_of_frame(event):
    """Handle person going out of frame."""
    print(f"🚫 OUT OF FRAME: {event['details']}")
    # High priority - avoiding detection

# 5. Performance monitoring
def monitor_engine_performance(engine):
    """Monitor and log engine performance."""
    stats = engine.get_statistics()
    
    print(f"📊 Performance Metrics:")
    print(f"   Frames processed: {stats['frame_count']}")
    print(f"   Average FPS: {stats['performance']['avg_fps']:.1f}")
    print(f"   Detection time: {stats['performance']['avg_detection_time_ms']:.1f}ms")
    print(f"   Active persons tracked: {stats['rule_engine']['active_persons']}")
    
    # Performance alerts
    if stats['performance']['avg_fps'] < 25:
        print("⚠️ Performance warning: FPS below 25")
    
    if stats['performance']['avg_detection_time_ms'] > 100:
        print("⚠️ Performance warning: Detection time above 100ms")

# 6. Integration with existing database
def integrate_with_database(engine, event):
    """Example of integrating events with existing database."""
    
    # Use the existing DBManager from the engine
    db = engine.db
    
    # Store event (if you want to extend the existing schema)
    try:
        # This would require adding to your existing database schema
        # db.store_research_event(
        #     session_id=engine.current_session_id,
        #     timestamp=event['timestamp'],
        #     person_id=event['person_id'],
        #     event_type=event['event_type'],
        #     severity=event['severity'],
        #     confidence=event['confidence'],
        #     details=event['details'],
        #     rule_triggered=event.get('rule_triggered'),
        #     bbox=event['bbox']
        # )
        pass
    except Exception as e:
        print(f"Database integration error: {e}")

if __name__ == "__main__":
    print("🚀 CheatGPT Research-Based Engine Integration Guide")
    print("This file shows examples of how to integrate the new engine.")
    print("Uncomment and modify the code above to fit your specific needs.")
"""
Debug Temporal Cheating Detection
=================================

Debug script to see what's happening with the temporal analysis during webcam testing.
This will help identify why sustained looking+leaning isn't escalating to cheating.
"""

import cv2
import numpy as np
import time
import logging
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging to see debug info
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

try:
    from cheatgpt.engine import Engine
    print("✅ Engine imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import engine: {e}")
    sys.exit(1)

class TemporalDebugTester:
    def __init__(self):
        """Initialize debug tester."""
        print("🔍 Initializing Temporal Debug Tester...")
        
        # Initialize engine with debug logging
        self.engine = Engine()
        self.camera_id = 0
        
        # Add temporal debugging
        self.frame_count = 0
        self.last_temporal_log = 0
        
    def initialize_camera(self):
        """Initialize camera."""
        print(f"📷 Initializing camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera initialized: {width}x{height} @ {fps}fps")
        return True
    
    def debug_frame_processing(self, frame):
        """Process frame and debug temporal analysis."""
        timestamp = time.time()
        
        try:
            # Process frame through engine
            overlay_frame, events = self.engine.process_frame(
                frame=frame,
                cam_id="debug_test",
                ts=timestamp
            )
            
            # Check temporal analysis state every 5 seconds
            if timestamp - self.last_temporal_log > 5.0:
                self.log_temporal_state()
                self.last_temporal_log = timestamp
            
            # Log any events
            if events:
                for event in events:
                    event_type = event.get('event_type', 'Unknown')
                    severity = event.get('severity', 'Unknown')
                    source = event.get('source', 'Unknown')
                    print(f"🚨 EVENT: {event_type} | {severity} | {source}")
                    
                    if source == 'temporal_analysis':
                        print(f"   🕐 TEMPORAL EVENT: {event.get('details', '')}")
            
            return overlay_frame, events
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return frame, []
    
    def log_temporal_state(self):
        """Log current temporal analysis state."""
        engine = self.engine
        
        print(f"\n📊 Temporal State @ Frame {self.frame_count}:")
        print(f"   Looking history: {len(engine.looking_around_history)} detections")
        print(f"   Leaning history: {len(engine.leaning_history)} detections")
        print(f"   Combined behavior start: {engine.combined_behavior_start}")
        print(f"   Cheating cooldown: {len(engine.cheating_cooldown)} persons")
        
        if engine.looking_around_history:
            recent_looking = max(engine.looking_around_history)
            looking_age = time.time() - recent_looking
            print(f"   Last looking detection: {looking_age:.1f}s ago")
        
        if engine.leaning_history:
            recent_leaning = max(engine.leaning_history)
            leaning_age = time.time() - recent_leaning
            print(f"   Last leaning detection: {leaning_age:.1f}s ago")
        
        print()
    
    def run_debug_test(self):
        """Run debug test with webcam."""
        if not self.initialize_camera():
            return
        
        print("\n🔍 Starting Temporal Debug Test")
        print("=" * 50)
        print("📹 Try this test sequence:")
        print("1. Sit normally for a few seconds")
        print("2. Start looking around (turn head left/right)")
        print("3. Also start leaning to one side")
        print("4. MAINTAIN both behaviors for 15+ seconds")
        print("5. Watch for temporal escalation to cheating")
        print("\n🔧 Press 'q' to quit, 's' to show state\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror effect
                self.frame_count += 1
                
                # Debug frame processing
                overlay_frame, events = self.debug_frame_processing(frame)
                
                # Add debug info to frame
                cv2.putText(overlay_frame, f"Frame: {self.frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay_frame, "Temporal Debug Mode", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display
                cv2.imshow('Temporal Debug Test', overlay_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.log_temporal_state()
                    
        except KeyboardInterrupt:
            print("\n🛑 Debug test interrupted")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Debug test complete")

def main():
    """Main function."""
    print("🔍 CheatGPT3 Temporal Debug Tester")
    print("=" * 40)
    
    # Check camera
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("❌ No webcam detected!")
        return
    test_cap.release()
    
    # Run debug test
    tester = TemporalDebugTester()
    tester.run_debug_test()

if __name__ == "__main__":
    main()

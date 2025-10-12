#!/usr/bin/env python3
"""
Test phone alarm functionality
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

from cheatgpt.engines.engine_hybrid import EngineHybrid
import time

def test_phone_alarm():
    """Test that phone detection triggers alarm"""
    
    print("🚨 Testing phone alarm functionality...")
    
    # Initialize engine
    engine = EngineHybrid()
    
    # Test phone detection with alarm
    print("\n--- Testing phone detection alarm ---")
    
    # Create mock frame (small numpy array)
    import numpy as np
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create phone detection that would trigger alarm
    # Process multiple frames to meet consecutive frame threshold
    for i in range(5):  # Process 5 frames to exceed threshold
        print(f"Processing frame {i+1} with phone detection...")
        
        # Mock detections - inject phone directly into engine
        engine.last_detections = [
            {
                'bbox': [300, 200, 350, 280],  # Mock phone bbox
                'conf': 0.8,
                'cls': 67,  # Phone class in YOLO
                'type': 'phone'
            }
        ]
        
        # Process frame
        overlay_frame, events = engine.process_frame(mock_frame, ts=float(i))
        
        if events:
            print(f"Events generated: {[event.get('event_type') for event in events]}")
            for event in events:
                if event.get('event_type') == 'Phone Usage':
                    print(f"✅ Phone usage event detected!")
                    print(f"📱 Details: {event.get('details')}")
                    print(f"🚨 Alarm should have triggered!")
        
        time.sleep(0.1)  # Small delay between frames
    
    print("\n🎯 Phone alarm test completed!")
    print("Check console for alarm messages and listen for sound")

if __name__ == "__main__":
    test_phone_alarm()
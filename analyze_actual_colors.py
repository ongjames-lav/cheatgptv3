#!/usr/bin/env python3
"""
Color Analysis Tool - Check actual colors produced by each event type
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ['PYTHONPATH'] = os.path.dirname(__file__)

from cheatgpt.engines.engine_hybrid import EngineHybrid

def analyze_actual_colors():
    """Analyze the actual colors produced by each event type."""
    print("🎨 ANALYZING ACTUAL COLORS PRODUCED")
    print("=" * 50)
    
    try:
        engine = EngineHybrid()
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_frame.fill(50)
        
        test_person = {
            'bbox': [300, 200, 600, 500],
            'confidence': 0.85,
            'person_id': 0
        }
        
        # Test different event types
        events_to_test = [
            {
                'name': 'Hand Gesture',
                'event': {
                    'person_id': 'person_000',
                    'event_type': 'Suspicious Hand Activity',
                    'confidence': 0.75,
                    'timestamp': time.time(),
                    'gesture_type': 'left_wrist_extended_arm_absolute',
                    'severity': 'orange'
                }
            },
            {
                'name': 'Sustained Head Turning', 
                'event': {
                    'person_id': 'person_000',
                    'event_type': 'Sustained Head Turning',
                    'confidence': 0.8,
                    'timestamp': time.time(),
                    'severity': 'red'
                }
            },
            {
                'name': 'Frequent Head Turning',
                'event': {
                    'person_id': 'person_000',
                    'event_type': 'Frequent Head Turning',
                    'confidence': 0.7,
                    'timestamp': time.time(),
                    'severity': 'orange'
                }
            },
            {
                'name': 'Phone Usage',
                'event': {
                    'person_id': 'person_000',
                    'event_type': 'Phone Usage Detected',
                    'confidence': 0.9,
                    'timestamp': time.time(),
                    'severity': 'orange'
                }
            },
            {
                'name': 'Looking Down',
                'event': {
                    'person_id': 'person_000',
                    'event_type': 'Abnormal Looking Down',
                    'confidence': 0.6,
                    'timestamp': time.time(),
                    'severity': 'yellow'
                }
            }
        ]
        
        for test_case in events_to_test:
            name = test_case['name']
            event = test_case['event']
            
            print(f"\n🔍 Testing {name}:")
            
            # Create overlay
            overlay_frame = engine._create_overlay(test_frame, [test_person], [event])
            
            # Analyze colors in bounding box
            bbox_area = overlay_frame[200:500, 300:600]
            
            # Get unique colors and their frequencies
            unique_colors = np.unique(bbox_area.reshape(-1, 3), axis=0)
            color_counts = []
            
            for color in unique_colors:
                count = np.sum(np.all(bbox_area == color, axis=2))
                color_counts.append((tuple(color), count))
            
            # Sort by frequency
            color_counts.sort(key=lambda x: x[1], reverse=True)
            
            # Show top colors
            print(f"   Top colors found:")
            for i, (color, count) in enumerate(color_counts[:5]):
                percentage = (count / bbox_area.size) * 100
                print(f"   {i+1}. BGR{color} - {count} pixels ({percentage:.1f}%)")
            
            # Check if it matches expected patterns
            dominant_color = color_counts[0][0] if color_counts else (0, 0, 0)
            b, g, r = dominant_color
            
            if name == 'Hand Gesture':
                is_magenta = (b > 100 and g < 100 and r > 100)
                print(f"   Is magenta-like? {'✅ YES' if is_magenta else '❌ NO'}")
            elif 'Head Turning' in name and event['severity'] == 'red':
                is_red = (b < 50 and g < 50 and r > 100)
                print(f"   Is red-like? {'✅ YES' if is_red else '❌ NO'}")
            elif event.get('severity') == 'orange':
                is_orange = (b < 50 and g > 80 and g < 220 and r > 150)
                print(f"   Is orange-like? {'✅ YES' if is_orange else '❌ NO'}")
            elif event.get('severity') == 'yellow':
                is_yellow = (b < 50 and g > 150 and r > 150)
                print(f"   Is yellow-like? {'✅ YES' if is_yellow else '❌ NO'}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_actual_colors()
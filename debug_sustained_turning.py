#!/usr/bin/env python3
"""
Debug Sustained Head Turning Detection
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ['PYTHONPATH'] = os.path.dirname(__file__)

from cheatgpt.engines.engine_hybrid import EngineHybrid

def test_sustained_turning():
    """Test sustained head turning detection logic."""
    print("TESTING SUSTAINED HEAD TURNING LOGIC")
    print("=" * 50)
    
    try:
        engine = EngineHybrid()
        
        # Get the thresholds
        print("Thresholds:")
        print(f"   Detection FPS: {engine.detection_fps}")
        print(f"   Sustained threshold: {engine.thresholds['head_turn_sustained_threshold']}s")
        print(f"   Frequent threshold: {engine.thresholds['head_turn_frequency_threshold']} turns")
        print(f"   Confirmation threshold: {engine.confirmation_threshold}")
        
        # Calculate minimum consecutive frames needed
        min_consecutive = engine.thresholds['head_turn_sustained_threshold'] * engine.detection_fps
        print(f"   Minimum consecutive frames needed: {min_consecutive}")
        
        # Test different scenarios
        scenarios = [
            {
                'name': 'No head turning',
                'events': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                'name': 'Frequent but not sustained (scattered)',
                'events': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            },
            {
                'name': 'Sustained at end (should trigger SUSTAINED)',
                'events': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 6 consecutive at end = 0.6s (less than 2.0s)
            },
            {
                'name': 'Long sustained at end (should trigger SUSTAINED)',
                'events': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 19 consecutive = 1.9s
            },
            {
                'name': 'Very long sustained (should definitely trigger)',
                'events': [0, 0] + [1] * 25  # 25 consecutive = 2.5s > 2.0s threshold
            }
        ]
        
        for scenario in scenarios:
            name = scenario['name']
            head_turn_events = scenario['events']
            
            print(f"\nScenario: {name}")
            print(f"   Events: {head_turn_events}")
            
            # Calculate consecutive turns at end
            consecutive_turns = 0
            for turn in reversed(head_turn_events):
                if turn:
                    consecutive_turns += 1
                else:
                    break
            
            sustained_duration = consecutive_turns / engine.detection_fps
            turn_count = sum(head_turn_events)
            
            print(f"   Total turns: {turn_count}")
            print(f"   Consecutive at end: {consecutive_turns}")
            print(f"   Sustained duration: {sustained_duration:.1f}s")
            
            # Check conditions
            frequent_turns = turn_count >= engine.thresholds['head_turn_frequency_threshold']
            sustained_turns = sustained_duration >= engine.thresholds['head_turn_sustained_threshold']
            
            print(f"   Frequent condition: {frequent_turns} (need {engine.thresholds['head_turn_frequency_threshold']}, have {turn_count})")
            print(f"   Sustained condition: {sustained_turns} (need {engine.thresholds['head_turn_sustained_threshold']}s, have {sustained_duration:.1f}s)")
            
            if sustained_turns:
                print(f"   PASS: Would trigger: SUSTAINED HEAD TURNING (RED)")
            elif frequent_turns:
                print(f"   PASS: Would trigger: FREQUENT HEAD TURNING (ORANGE)")
            else:
                print(f"   FAIL: Would NOT trigger")
        
        # Test real-time scenario simulation
        print(f"\nREAL-TIME SIMULATION:")
        print(f"   Simulating continuous head turning for 3 seconds...")
        
        # Simulate 30 frames (3 seconds at 10 FPS)
        frames_needed = int(3.0 * engine.detection_fps)
        continuous_events = [1] * frames_needed
        
        consecutive_turns = 0
        for turn in reversed(continuous_events):
            if turn:
                consecutive_turns += 1
            else:
                break
        
        sustained_duration = consecutive_turns / engine.detection_fps
        turn_count = sum(continuous_events)
        
        print(f"   Frames: {frames_needed}")
        print(f"   Events: {continuous_events}")
        print(f"   Consecutive at end: {consecutive_turns}")
        print(f"   Duration: {sustained_duration:.1f}s")
        
        sustained_turns = sustained_duration >= engine.thresholds['head_turn_sustained_threshold']
        print(f"   Should trigger sustained: {sustained_turns}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_sustained_turning()
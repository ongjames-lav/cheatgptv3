"""
Test Temporal Cheating Detection
================================

Test the new sustained behavior detection that identifies cheating when
both "looking around" and "leaning" persist for 10-15 seconds.

This demonstrates how the system escalates from individual behavior alerts
to serious cheating detection when patterns persist over time.
"""

import time
import logging
from cheatgpt.engine import Engine

# Configure logging to see temporal analysis
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_temporal_cheating_detection():
    """Test the temporal cheating detection system."""
    
    print("🕐 Testing Temporal Cheating Detection")
    print("=" * 50)
    
    # Initialize engine
    engine = Engine()
    
    print(f"✅ Engine initialized")
    print(f"   Temporal cheating enabled: {engine.temporal_cheating_enabled}")
    print(f"   Threshold: {engine.temporal_cheating_threshold} seconds")
    print(f"   Cooldown duration: {engine.cheating_cooldown_duration} seconds")
    print()
    
    print("📊 Simulating sustained suspicious behavior...")
    print("   This simulates a user who:")
    print("   - Continuously looks around (head turning)")
    print("   - Continuously leans to the side")
    print("   - Both behaviors persist for >12 seconds")
    print()
    
    # Simulate sustained suspicious behavior
    simulation_duration = 20  # seconds
    frame_interval = 1.0 / 30.0  # 30 FPS
    total_frames = int(simulation_duration / frame_interval)
    
    start_time = time.time()
    events_detected = []
    
    print("🎬 Starting simulation...")
    print(f"   Duration: {simulation_duration} seconds ({total_frames} frames)")
    print("   Expected: Sustained cheating alert after ~12 seconds")
    print()
    
    for frame_num in range(total_frames):
        current_time = start_time + (frame_num * frame_interval)
        
        # Simulate continuous suspicious behavior
        # Both looking and leaning flags are true for entire duration
        pose_results = [{
            'person_id': 'test_person_001',
            'lean_flag': True,      # Continuous leaning
            'look_flag': True,      # Continuous looking around
            'phone_flag': False,
            'lean_angle': 15.0,     # Significant lean
            'head_turn_angle': -25.0,  # Significant head turn
            'confidence': 0.9,
            'bbox': [100, 100, 200, 300]
        }]
        
        # Track temporal behaviors
        temporal_events = engine._track_temporal_behaviors(pose_results, current_time)
        
        # Check for cheating detection
        if temporal_events:
            for event in temporal_events:
                if event['event_type'] == 'Sustained Cheating Behavior':
                    events_detected.append({
                        'frame': frame_num,
                        'time': current_time - start_time,
                        'event': event
                    })
                    print(f"🚨 CHEATING DETECTED at frame {frame_num} ({current_time - start_time:.1f}s)")
                    print(f"   Overlap duration: {event['overlap_duration']:.1f}s")
                    print(f"   Looking duration: {event['looking_duration']:.1f}s") 
                    print(f"   Leaning duration: {event['leaning_duration']:.1f}s")
                    print()
        
        # Print progress every 5 seconds
        elapsed = current_time - start_time
        if frame_num % 150 == 0:  # Every ~5 seconds at 30fps
            print(f"⏱️  Elapsed: {elapsed:.1f}s - Behavior continuing...")
    
    print("\n📊 Simulation Results:")
    print("=" * 50)
    print(f"Total cheating events detected: {len(events_detected)}")
    
    if events_detected:
        first_detection = events_detected[0]
        print(f"First detection at: {first_detection['time']:.1f} seconds")
        print(f"Detection details:")
        event = first_detection['event']
        print(f"  - Severity: {event['severity']}")
        print(f"  - Confidence: {event['confidence']}")
        print(f"  - Source: {event['source']}")
        print(f"  - Details: {event['details']}")
        
        # Test cooldown system
        print(f"\nCooldown active: {'test_person_001' in engine.cheating_cooldown}")
        if 'test_person_001' in engine.cheating_cooldown:
            cooldown_time = engine.cheating_cooldown['test_person_001']
            print(f"Cooldown until: {time.time() - cooldown_time:.1f}s ago")
    else:
        print("❌ No sustained cheating detected (check threshold settings)")
    
    print("\n✅ Temporal cheating detection test complete!")

def test_configuration():
    """Test different configuration values."""
    print("\n🔧 Testing Configuration Options")
    print("=" * 40)
    
    engine = Engine()
    
    print("Current settings:")
    print(f"  Temporal threshold: {engine.temporal_cheating_threshold}s")
    print(f"  Required frames: {engine.required_frames}")
    print(f"  FPS estimate: {engine.fps_estimate}")
    print(f"  Cooldown duration: {engine.cheating_cooldown_duration}s")
    
    print("\nBehavior analysis:")
    print("  - Brief movements (<12s): Normal alerts only") 
    print("  - Sustained behavior (≥12s): Escalated to cheating")
    print("  - Cooldown prevents alert spam for 30s")
    
def main():
    """Main test function."""
    print("🧠 CheatGPT3 Temporal Cheating Detection Test")
    print("=" * 60)
    print()
    
    test_configuration()
    test_temporal_cheating_detection()
    
    print("\n🎯 How it works in real webcam testing:")
    print("1. System tracks 'looking around' and 'leaning' detections over time")
    print("2. When BOTH behaviors persist for ≥12 seconds continuously:")
    print("   → Escalates from yellow/orange alerts to RED cheating alert")
    print("3. Cooldown prevents spam alerts for same person")
    print("4. This catches sustained suspicious patterns vs brief movements")

if __name__ == "__main__":
    main()

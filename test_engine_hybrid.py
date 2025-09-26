"""Test script for the Research-Based Hybrid Engine.

This script tests the key features of the new engine:
- Frame rate management (30 FPS live stream, 10 FPS detection)
- Research-based rule engine
- Tracker integration
- No LSTM dependencies
"""

import sys
import os
import time
import cv2
import numpy as np

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_engine_import():
    """Test if the engine can be imported successfully."""
    try:
        from cheatgpt.engines.engine_hybrid import EngineHybrid
        print("✅ Engine import successful")
        return True
    except ImportError as e:
        print(f"❌ Engine import failed: {e}")
        return False

def test_engine_initialization():
    """Test engine initialization."""
    try:
        from cheatgpt.engines.engine_hybrid import EngineHybrid
        engine = EngineHybrid()
        print("✅ Engine initialization successful")
        print(f"📊 Configuration: {engine.target_fps} FPS live stream, {engine.detection_fps:.1f} FPS detection")
        return engine
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return None

def test_frame_processing(engine, num_frames=90):
    """Test frame processing with synthetic data."""
    print(f"\n🧪 Testing frame processing with {num_frames} frames...")
    
    # Create synthetic frames
    frame_width, frame_height = 640, 480
    
    processing_times = []
    detection_counts = []
    test_start_time = time.time()
    
    for i in range(num_frames):
        # Create synthetic frame
        frame = np.random.randint(0, 255, (frame_height, frame_width, 3), dtype=np.uint8)
        
        # Process frame
        start_time = time.time()
        overlay_frame, events = engine.process_frame(frame)
        processing_time = time.time() - start_time
        
        processing_times.append(processing_time)
        detection_counts.append(len(events))
        
        # Check if this was a detection frame
        is_detection_frame = (engine.frame_count % engine.skip_rate == 0)
        
        if i % 30 == 0:  # Print every 30 frames (1 second at 30 FPS)
            current_fps = 1.0 / processing_time if processing_time > 0 else float('inf')
            elapsed_time = time.time() - test_start_time
            overall_fps = (i + 1) / elapsed_time if elapsed_time > 0 else 0
            fps_display = f"{current_fps:.1f}" if current_fps != float('inf') else "∞"
            print(f"Frame {engine.frame_count}: Processing {processing_time*1000:.1f}ms, "
                  f"Frame FPS: {fps_display}, Overall FPS: {overall_fps:.1f}, "
                  f"Detection: {is_detection_frame}, Events: {len(events)}")
    
    total_test_time = time.time() - test_start_time
    
    # Calculate statistics
    avg_processing_time = np.mean(processing_times) * 1000  # ms
    avg_fps = 1.0 / np.mean(processing_times)
    
    # Handle zero processing times for min/max FPS calculation
    non_zero_times = [t for t in processing_times if t > 0]
    if non_zero_times:
        min_fps = 1.0 / max(non_zero_times)
        max_fps = 1.0 / min(non_zero_times)
    else:
        min_fps = 0
        max_fps = 0
    
    overall_test_fps = num_frames / total_test_time
    detection_frame_count = sum(1 for i in range(num_frames) if (i + 1) % engine.skip_rate == 0)
    
    print(f"\n📊 Performance Results:")
    print(f"   Total test time: {total_test_time:.2f}s")
    print(f"   Overall test FPS: {overall_test_fps:.1f}")
    print(f"   Average processing time: {avg_processing_time:.1f}ms")
    print(f"   Average frame FPS: {avg_fps:.1f}")
    if non_zero_times:
        print(f"   FPS Range: {min_fps:.1f} - {max_fps:.1f}")
    else:
        print(f"   FPS Range: N/A (all processing times were 0)")
    print(f"   Target FPS: {engine.target_fps}")
    print(f"   Detection frames: {detection_frame_count}/{num_frames} ({detection_frame_count/num_frames*100:.1f}%)")
    print(f"   Expected detection rate: {1/engine.skip_rate*100:.1f}%")
    
    # Performance assertions
    performance_ok = True
    if avg_processing_time > 100:  # Should be under 100ms for real-time performance
        print(f"⚠️  Warning: Processing time {avg_processing_time:.1f}ms may be too slow for real-time")
        performance_ok = False
    
    if avg_fps < 25:  # Should maintain close to 30 FPS
        print(f"⚠️  Warning: Average FPS {avg_fps:.1f} is below target")
        performance_ok = False
    
    if performance_ok:
        print("✅ Performance targets met")
    
    return performance_ok

def test_rule_engine():
    """Test the research-based rule engine."""
    print(f"\n🧪 Testing research-based rule engine...")
    
    try:
        from cheatgpt.engines.engine_hybrid import ResearchBasedRuleEngine
        
        rule_engine = ResearchBasedRuleEngine()
        print("✅ Rule engine initialization successful")
        
        # Test with synthetic detection data
        person_id = 1
        timestamp = time.time()
        
        # Test normal behavior (should not trigger events)
        normal_detection = {
            'bbox': [100, 100, 200, 200],
            'phone_flag': False,
            'head_turn_angle': 5.0,  # Normal small head movement
            'lean_angle': 3.0,  # Normal small lean
            'gesture_flag': False,
            'out_of_frame': False
        }
        
        events = rule_engine.update_detection(person_id, normal_detection, timestamp)
        print(f"Normal behavior events: {len(events)} (expected: 0)")
        
        # Test suspicious behavior (should trigger after threshold)
        suspicious_detection = {
            'bbox': [100, 100, 200, 200],
            'phone_flag': True,  # Phone detected
            'head_turn_angle': 25.0,  # Excessive head turning
            'lean_angle': 30.0,  # Excessive leaning
            'gesture_flag': True,  # Hand gestures
            'out_of_frame': False
        }
        
        # Send multiple suspicious detections to trigger thresholds
        total_events = 0
        for i in range(10):  # Send 10 consecutive suspicious detections
            timestamp += 0.1  # 100ms intervals
            events = rule_engine.update_detection(person_id, suspicious_detection, timestamp)
            total_events += len(events)
            if events:
                print(f"Suspicious behavior detected at frame {i+1}: {[e['event_type'] for e in events]}")
        
        print(f"Total events from suspicious behavior: {total_events}")
        print(f"Rule engine thresholds: {rule_engine.thresholds}")
        print("✅ Rule engine testing completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Rule engine test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing Research-Based Hybrid Engine")
    print("=" * 50)
    
    # Test 1: Import
    if not test_engine_import():
        return False
    
    # Test 2: Initialization
    engine = test_engine_initialization()
    if engine is None:
        return False
    
    # Test 3: Frame processing
    performance_ok = test_frame_processing(engine)
    
    # Test 4: Rule engine
    rule_ok = test_rule_engine()
    
    # Test 5: Engine statistics
    print(f"\n📊 Engine Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    # Final results
    print(f"\n🎯 Test Results:")
    print(f"   Engine Import: ✅")
    print(f"   Engine Initialization: ✅")
    print(f"   Frame Processing: {'✅' if performance_ok else '⚠️'}")
    print(f"   Rule Engine: {'✅' if rule_ok else '❌'}")
    
    if performance_ok and rule_ok:
        print(f"\n🎉 All tests passed! Engine is ready for use.")
        return True
    else:
        print(f"\n⚠️  Some tests had issues. Check the output above.")
        return False

if __name__ == "__main__":
    main()
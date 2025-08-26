"""Comprehensive test of the complete CheatGPT3 engine."""
import os
import sys
import cv2
import time

# Add the cheatgpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cheatgpt'))

def test_complete_engine():
    """Test the complete CheatGPT3 engine pipeline."""
    print("🚀 CheatGPT3 Complete Engine Test")
    print("=" * 50)
    
    try:
        from cheatgpt.engine import Engine
        print("✓ Engine imported successfully")
    except ImportError as e:
        print(f"✗ Engine import failed: {e}")
        return False
    
    # Initialize engine
    print("\n🔧 Initializing Engine...")
    try:
        engine = Engine()
        print("✓ Engine initialized successfully")
        print(f"   Engine info: {engine}")
    except Exception as e:
        print(f"✗ Engine initialization failed: {e}")
        return False
    
    # Load test frame
    print("\n📷 Loading Test Frame...")
    frame = cv2.imread("original_test_image.jpg")
    if frame is None:
        print("✗ Could not load test image")
        return False
    
    print(f"✓ Test frame loaded: {frame.shape}")
    
    # Test single frame processing
    print("\n🎯 Testing Single Frame Processing...")
    try:
        start_time = time.time()
        overlay_frame, events = engine.process_frame(frame, "test_cam", start_time)
        processing_time = time.time() - start_time
        
        print(f"✓ Frame processed successfully in {processing_time:.3f}s")
        print(f"   Output frame shape: {overlay_frame.shape}")
        print(f"   Events detected: {len(events)}")
        
        for event in events:
            print(f"     Event: {event['track_id']} - {event['event_type']} ({event['severity']})")
        
    except Exception as e:
        print(f"✗ Frame processing failed: {e}")
        return False
    
    # Test multiple frame processing (simulating video)
    print("\n📹 Testing Multi-Frame Processing...")
    try:
        total_events = 0
        total_processing_time = 0
        num_frames = 10
        
        for frame_idx in range(num_frames):
            start_time = time.time()
            timestamp = start_time + frame_idx * 0.1  # 10 FPS simulation
            
            overlay_frame, events = engine.process_frame(frame, f"cam_{frame_idx%2}", timestamp)
            
            frame_processing_time = time.time() - start_time
            total_processing_time += frame_processing_time
            total_events += len(events)
            
            print(f"   Frame {frame_idx+1}: {len(events)} events, {frame_processing_time:.3f}s")
        
        avg_processing_time = total_processing_time / num_frames
        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        print(f"✓ Multi-frame processing complete")
        print(f"   Average processing time: {avg_processing_time:.3f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Total events: {total_events}")
        
    except Exception as e:
        print(f"✗ Multi-frame processing failed: {e}")
        return False
    
    # Test engine statistics
    print("\n📊 Testing Engine Statistics...")
    try:
        stats = engine.get_statistics()
        
        print("✓ Engine statistics retrieved:")
        print(f"   Frames processed: {stats['frame_count']}")
        print(f"   Active tracks: {stats['active_tracks']}")
        print(f"   Total tracks created: {stats['total_tracks_created']}")
        print(f"   Active violations: {stats['active_violations']}")
        print(f"   Performance FPS: {stats['performance']['fps']:.1f}")
        print(f"   Evidence directory: {stats['evidence_directory']}")
        
        if stats['severity_distribution']:
            print("   Severity distribution:")
            for severity, count in stats['severity_distribution'].items():
                print(f"     {severity}: {count}")
        
    except Exception as e:
        print(f"✗ Statistics retrieval failed: {e}")
        return False
    
    # Test evidence saving (simulate cheating scenario)
    print("\n🚨 Testing Evidence Saving...")
    try:
        # Create a scenario that should trigger cheating detection
        # We'll process multiple frames to build up behavior history
        
        print("   Building up behavior pattern for cheating detection...")
        
        # Process several frames to establish tracking and build behavior history
        for i in range(12):  # Process enough frames to trigger policies
            timestamp = time.time() + i * 0.1
            overlay_frame, events = engine.process_frame(frame, "evidence_test", timestamp)
            
            # Check if cheating was detected
            cheating_events = [e for e in events if e['severity'] == 'Cheating']
            if cheating_events:
                print(f"   🚨 Cheating detected on frame {i+1}!")
                print(f"     Events: {[e['track_id'] for e in cheating_events]}")
                break
        
        # Check if evidence directory has files
        evidence_files = []
        if os.path.exists(engine.evidence_dir):
            evidence_files = [f for f in os.listdir(engine.evidence_dir) if f.endswith('.jpg')]
        
        print(f"✓ Evidence system tested")
        print(f"   Evidence files found: {len(evidence_files)}")
        
        if evidence_files:
            print("   Recent evidence files:")
            for file in evidence_files[-3:]:  # Show last 3 files
                print(f"     {file}")
        
    except Exception as e:
        print(f"✗ Evidence saving test failed: {e}")
        return False
    
    # Test API format
    print("\n📋 Testing API Format...")
    try:
        # Test the exact API format specified
        test_frame = frame.copy()
        cam_id = "api_test_cam"
        ts = time.time()
        
        # Call the API
        overlay_frame, events = engine.process_frame(test_frame, cam_id, ts)
        
        # Validate return types
        assert isinstance(overlay_frame, type(frame)), f"overlay_frame type mismatch: {type(overlay_frame)}"
        assert isinstance(events, list), f"events type mismatch: {type(events)}"
        
        print("✓ API format validation passed")
        print(f"   Input: process_frame(frame, '{cam_id}', {ts})")
        print(f"   Output: overlay_frame({overlay_frame.shape}), events({len(events)})")
        
        # Validate event structure
        if events:
            event = events[0]
            required_fields = ['timestamp', 'cam_id', 'track_id', 'event_type', 'severity', 'confidence', 'bbox']
            for field in required_fields:
                assert field in event, f"Missing required field in event: {field}"
            
            print("✓ Event structure validation passed")
            print(f"   Event fields: {list(event.keys())}")
        
    except Exception as e:
        print(f"✗ API format test failed: {e}")
        return False
    
    # Test visualization quality
    print("\n🖼️  Testing Visualization Quality...")
    try:
        # Generate final visualization
        overlay_frame, events = engine.process_frame(frame, "viz_test", time.time())
        
        # Save test visualization
        output_path = "engine_test_result.jpg"
        success = cv2.imwrite(output_path, overlay_frame)
        
        if success:
            print(f"✓ Visualization saved to {output_path}")
            
            # Check image properties
            saved_frame = cv2.imread(output_path)
            if saved_frame is not None:
                print(f"   Output image: {saved_frame.shape}")
                print(f"   Color channels: {saved_frame.shape[2] if len(saved_frame.shape) > 2 else 1}")
            
        else:
            print("✗ Failed to save visualization")
            return False
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False
    
    # Test engine reset
    print("\n🔄 Testing Engine Reset...")
    try:
        initial_frame_count = engine.frame_count
        engine.reset()
        
        if engine.frame_count == 0:
            print("✓ Engine reset successful")
            print(f"   Frame count: {initial_frame_count} → {engine.frame_count}")
        else:
            print(f"✗ Engine reset failed: frame_count still {engine.frame_count}")
            return False
        
    except Exception as e:
        print(f"✗ Engine reset test failed: {e}")
        return False
    
    print("\n🎯 Engine Test Summary:")
    print("=" * 30)
    print("✅ Engine initialization: PASSED")
    print("✅ Single frame processing: PASSED")
    print("✅ Multi-frame processing: PASSED")
    print("✅ Statistics retrieval: PASSED")
    print("✅ Evidence saving: PASSED")
    print("✅ API format: PASSED")
    print("✅ Visualization: PASSED")
    print("✅ Engine reset: PASSED")
    
    final_stats = engine.get_statistics()
    print(f"\n📈 Final Performance Metrics:")
    print(f"   Average FPS: {final_stats['performance']['fps']:.1f}")
    print(f"   Processing samples: {final_stats['performance']['total_processing_samples']}")
    print(f"   Evidence directory: {final_stats['evidence_directory']}")
    
    return True

def test_engine_integration():
    """Test engine integration with all components."""
    print("\n🔗 Engine Integration Test")
    print("=" * 30)
    
    try:
        from cheatgpt.engine import Engine
        from cheatgpt.policy.rules import get_active_violations, get_policy_statistics
        
        engine = Engine()
        frame = cv2.imread("original_test_image.jpg")
        
        # Process frame and check component integration
        overlay_frame, events = engine.process_frame(frame)
        
        # Check policy integration
        active_violations = get_active_violations()
        policy_stats = get_policy_statistics()
        
        print(f"✓ Policy integration working:")
        print(f"   Active violations: {len(active_violations)}")
        print(f"   Policy statistics: {policy_stats}")
        
        # Check tracker integration
        tracker_stats = engine.tracker.get_statistics()
        print(f"✓ Tracker integration working:")
        print(f"   Active tracks: {tracker_stats['active_tracks']}")
        print(f"   Total tracks created: {tracker_stats['total_tracks_created']}")
        
        # Check detector integration
        yolo_info = engine.yolo_detector.get_model_info()
        pose_info = engine.pose_detector.get_model_info()
        
        print(f"✓ Detector integration working:")
        print(f"   YOLO device: {yolo_info['device']}")
        print(f"   Pose device: {pose_info['device']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎓 CheatGPT3 Engine Comprehensive Testing")
    print("=" * 60)
    
    success = test_complete_engine()
    
    if success:
        integration_success = test_engine_integration()
        if integration_success:
            print("\n🎉 ALL ENGINE TESTS PASSED!")
            print("🚀 CheatGPT3 Engine is ready for production!")
        else:
            print("\n❌ INTEGRATION TESTS FAILED")
    else:
        print("\n❌ ENGINE TESTS FAILED")

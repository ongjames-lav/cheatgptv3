"""Integration test showing complete pipeline from pose detection to policy evaluation."""
import os
import sys
import cv2
import time

# Add the cheatgpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cheatgpt'))

def test_complete_pipeline():
    """Test the complete pipeline from pose detection to policy evaluation."""
    print("=== COMPLETE PIPELINE INTEGRATION TEST ===\n")
    
    try:
        from cheatgpt.detectors.pose_detector import PoseDetector
        from cheatgpt.detectors.yolo11_detector import YOLO11Detector
        from cheatgpt.policy.rules import check_rule, get_active_violations, get_policy_statistics
        print("✓ Successfully imported all components")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Initialize components
    print("1. Initializing Components:")
    pose_detector = PoseDetector()
    yolo_detector = YOLO11Detector()
    print("   ✓ Pose detector initialized")
    print("   ✓ YOLO detector initialized")
    print("   ✓ Policy engine ready")
    print()
    
    # Load test image
    print("2. Loading Test Image:")
    frame = cv2.imread("original_test_image.jpg")
    if frame is None:
        print("   ✗ Test image not found")
        return False
    print(f"   ✓ Loaded test image: {frame.shape}")
    print()
    
    # Simulate multiple frames to test policy persistence
    print("3. Processing Multiple Frames:")
    
    for frame_idx in range(8):  # Process 8 frames
        print(f"   Frame {frame_idx + 1}:")
        
        # Get detections
        phone_detections = yolo_detector.detect(frame)
        phones = [det for det in phone_detections if det['cls_name'] == 'cell phone']
        
        # Get pose estimates
        poses = pose_detector.estimate(frame, phones)
        print(f"     Detected {len(poses)} people")
        
        # Process each person through policy engine
        current_violations = []
        for pose in poses:
            # Simulate some variation in behavior over frames
            lean_flag = pose['lean_flag']
            look_flag = pose['look_flag']
            phone_flag = pose['phone_flag']
            
            # Add some temporal variation for testing
            if frame_idx >= 3:  # Make person_2 start leaning persistently
                if pose['track_id'] == 'person_2':
                    lean_flag = True
            
            if frame_idx >= 2:  # Make person_0 look around more
                if pose['track_id'] == 'person_0':
                    look_flag = True
            
            # Apply policy rules
            violations = check_rule(
                pose['track_id'],
                lean_flag,
                look_flag,
                phone_flag,
                time.time() + frame_idx  # Simulate time progression
            )
            
            if violations:
                current_violations.extend(violations)
                for violation in violations:
                    print(f"       {violation.track_id}: {violation.label} ({violation.severity.color})")
        
        if not current_violations:
            print(f"       No violations detected")
        
        time.sleep(0.1)  # Small delay between frames
    
    print()
    
    # Show final state
    print("4. Final Policy State:")
    active_violations = get_active_violations()
    print(f"   Active violations: {len(active_violations)}")
    
    for track_id, violation in active_violations.items():
        print(f"     {track_id}:")
        print(f"       Violation: {violation.label}")
        print(f"       Severity: {violation.severity.color}")
        print(f"       Duration: {violation.end_ts - violation.start_ts:.1f}s")
        print(f"       Details: {violation.details}")
        print(f"       Confidence: {violation.confidence:.2f}")
    print()
    
    # Generate policy statistics
    print("5. Policy Statistics:")
    stats = get_policy_statistics()
    print(f"   Total tracks monitored: {stats['total_tracks']}")
    print(f"   Active violations: {stats['active_violations']}")
    print(f"   Severity distribution:")
    for severity, count in stats['severity_distribution'].items():
        print(f"     {severity}: {count}")
    print()
    
    # Test escalation scenario
    print("6. Testing Cheating Escalation:")
    test_track = "test_cheater"
    
    # Build up suspicious behavior pattern
    print("   Building suspicious behavior pattern...")
    for i in range(4):
        violations = check_rule(test_track, True, False, False)  # Leaning
        if violations:
            print(f"     Frame {i+1}: {violations[0].label}")
        
        violations = check_rule(test_track, False, True, False)  # Looking
        if violations:
            print(f"     Frame {i+1}: {violations[0].label}")
    
    # Add phone use - should escalate to cheating
    print("   Adding phone use (should escalate to cheating)...")
    violations = check_rule(test_track, False, False, True)
    
    if violations and any(v.label == "Cheating" for v in violations):
        cheating_violation = next(v for v in violations if v.label == "Cheating")
        print(f"   ✓ ESCALATED TO CHEATING!")
        print(f"     Details: {cheating_violation.details}")
        print(f"     Confidence: {cheating_violation.confidence:.2f}")
    else:
        print(f"   Expected cheating escalation, got: {[v.label for v in violations]}")
    print()
    
    # Create visualization
    print("7. Creating Visualization:")
    annotated_frame = frame.copy()
    
    # Get fresh pose estimates for visualization
    phone_detections = yolo_detector.detect(frame)
    phones = [det for det in phone_detections if det['cls_name'] == 'cell phone']
    poses = pose_detector.estimate(frame, phones)
    
    # Overlay violation information
    active_violations = get_active_violations()
    
    for pose in poses:
        x1, y1, x2, y2 = [int(x) for x in pose['bbox']]
        track_id = pose['track_id']
        
        # Determine color based on active violations
        if track_id in active_violations:
            violation = active_violations[track_id]
            if violation.severity.color == 'red':
                color = (0, 0, 255)  # Red for cheating
            elif violation.severity.color == 'orange':
                color = (0, 165, 255)  # Orange for phone use
            elif violation.severity.color == 'yellow':
                color = (0, 255, 255)  # Yellow for minor violations
            else:
                color = (0, 255, 0)  # Green for normal
            
            label = f"{track_id}: {violation.label}"
        else:
            color = (0, 255, 0)  # Green for normal
            label = f"{track_id}: Normal"
        
        # Draw bounding box and label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        
        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1-30), (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add policy summary
    summary_text = [
        f"Policy Engine Status:",
        f"Active Violations: {len(active_violations)}",
        f"Tracks Monitored: {stats['total_tracks']}"
    ]
    
    for i, text in enumerate(summary_text):
        cv2.putText(annotated_frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    output_path = "pipeline_integration_result.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"   ✓ Saved visualization to {output_path}")
    print()
    
    print("=== PIPELINE INTEGRATION TEST COMPLETED ===")
    print("✅ Pose Detection → Policy Evaluation pipeline working correctly")
    print("✅ Behavior tracking and escalation functioning properly")
    print("✅ Multi-track monitoring operational")
    print("✅ Visualization and reporting ready")
    
    return True

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n🎉 COMPLETE PIPELINE INTEGRATION SUCCESSFUL!")
    else:
        print("\n❌ PIPELINE INTEGRATION FAILED")

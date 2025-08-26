"""Demo script showing the complete CheatGPT3 detection pipeline."""
import os
import sys
import cv2
import time

# Add the cheatgpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cheatgpt'))

def demo_cheatgpt_pipeline():
    """Demonstrate the complete CheatGPT3 detection and policy pipeline."""
    print("🎓 CheatGPT3 - Complete Detection Pipeline Demo")
    print("=" * 50)
    
    try:
        from cheatgpt.detectors.pose_detector import PoseDetector
        from cheatgpt.detectors.yolo11_detector import YOLO11Detector
        from cheatgpt.policy.rules import (
            check_rule, get_active_violations, get_policy_statistics, Severity
        )
        print("✓ All components loaded successfully")
    except ImportError as e:
        print(f"✗ Component loading failed: {e}")
        return
    
    # Initialize detection pipeline
    print("\n🔧 Initializing Detection Components...")
    pose_detector = PoseDetector()
    yolo_detector = YOLO11Detector()
    print("✓ YOLO11 Object Detector ready")
    print("✓ YOLOv11-Pose Detector ready")  
    print("✓ Policy Engine ready")
    
    # Load exam room image
    print("\n📷 Loading Exam Room Scene...")
    frame = cv2.imread("original_test_image.jpg")
    if frame is None:
        print("✗ Could not load test image")
        return
    
    print(f"✓ Exam room image loaded: {frame.shape}")
    print("✓ Ready to monitor students")
    
    # Simulate real-time monitoring
    print("\n🎯 Starting Student Monitoring...")
    print("   Processing frames to detect suspicious behavior...")
    
    for frame_num in range(1, 11):  # Simulate 10 frames
        print(f"\n   📋 Frame {frame_num}:")
        
        # Step 1: Object Detection
        detections = yolo_detector.detect(frame)
        phones = [d for d in detections if d['cls_name'] == 'cell phone']
        people = [d for d in detections if d['cls_name'] == 'person']
        
        print(f"      📱 Detected {len(phones)} phones, {len(people)} people")
        
        # Step 2: Pose Analysis
        poses = pose_detector.estimate(frame, phones)
        print(f"      🤸 Analyzed poses for {len(poses)} students")
        
        # Step 3: Behavior Analysis & Policy Evaluation
        frame_violations = []
        
        for pose in poses:
            # Simulate some behavioral variation over time
            lean_flag = pose['lean_flag']
            look_flag = pose['look_flag'] 
            phone_flag = pose['phone_flag']
            
            # Add temporal patterns for demo
            if frame_num >= 5 and pose['track_id'] == 'person_0':
                look_flag = True  # Student starts looking around
            if frame_num >= 7 and pose['track_id'] == 'person_2':
                lean_flag = True  # Student starts leaning
            if frame_num >= 9 and pose['track_id'] == 'person_0':
                phone_flag = True  # Student uses phone (potential cheating)
            
            # Apply policy rules
            violations = check_rule(
                pose['track_id'],
                lean_flag,
                look_flag,
                phone_flag,
                time.time() + frame_num  # Simulate time progression
            )
            
            if violations:
                frame_violations.extend(violations)
        
        # Report frame results
        if frame_violations:
            for violation in frame_violations:
                severity_icon = {
                    'yellow': '⚠️',
                    'orange': '🔶', 
                    'red': '🚨'
                }.get(violation.severity.color, 'ℹ️')
                
                print(f"      {severity_icon} {violation.track_id}: {violation.label}")
        else:
            print(f"      ✅ All students behaving normally")
        
        time.sleep(0.2)  # Simulate processing time
    
    # Final monitoring report
    print(f"\n📊 Final Monitoring Report:")
    print("=" * 30)
    
    active_violations = get_active_violations()
    stats = get_policy_statistics()
    
    print(f"👥 Students Monitored: {stats['total_tracks']}")
    print(f"🚨 Active Violations: {stats['active_violations']}")
    
    if active_violations:
        print(f"\n⚠️  Current Violations:")
        for track_id, violation in active_violations.items():
            severity_color = {
                'green': '🟢',
                'yellow': '🟡', 
                'orange': '🟠',
                'red': '🔴'
            }.get(violation.severity.color, '⚪')
            
            duration = violation.end_ts - violation.start_ts
            print(f"   {severity_color} {track_id}: {violation.label}")
            print(f"      Duration: {duration:.1f}s | Confidence: {violation.confidence:.0%}")
            print(f"      Details: {violation.details}")
    
    # Severity distribution
    if stats['severity_distribution']:
        print(f"\n📈 Violation Distribution:")
        for severity, count in stats['severity_distribution'].items():
            print(f"   {severity}: {count} students")
    
    # Create annotated visualization
    print(f"\n🖼️  Creating Annotated Visualization...")
    
    # Get final detection state
    detections = yolo_detector.detect(frame)
    phones = [d for d in detections if d['cls_name'] == 'cell phone']
    poses = pose_detector.estimate(frame, phones)
    
    # Create visualization
    annotated_frame = frame.copy()
    active_violations = get_active_violations()
    
    for pose in poses:
        x1, y1, x2, y2 = [int(coord) for coord in pose['bbox']]
        track_id = pose['track_id']
        
        # Determine status and color
        if track_id in active_violations:
            violation = active_violations[track_id]
            color_map = {
                'green': (0, 255, 0),
                'yellow': (0, 255, 255),
                'orange': (0, 165, 255),
                'red': (0, 0, 255)
            }
            color = color_map.get(violation.severity.color, (128, 128, 128))
            status = violation.label
        else:
            color = (0, 255, 0)  # Green for normal
            status = "Normal"
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        
        # Add status label
        label = f"{track_id}: {status}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Label background
        cv2.rectangle(annotated_frame, (x1, y1-35), (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1+5, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add system status overlay
    status_lines = [
        "CheatGPT3 - Exam Monitoring System",
        f"Status: {'VIOLATIONS DETECTED' if active_violations else 'ALL CLEAR'}",
        f"Students: {stats['total_tracks']} | Violations: {stats['active_violations']}"
    ]
    
    for i, line in enumerate(status_lines):
        color = (0, 0, 255) if active_violations and i == 1 else (255, 255, 255)
        cv2.putText(annotated_frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Save result
    output_path = "cheatgpt_demo_result.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"✓ Saved monitoring result to {output_path}")
    
    # Summary
    print(f"\n🎯 Demo Summary:")
    print("=" * 20)
    print("✅ Object Detection: Successfully detected people and phones")
    print("✅ Pose Analysis: Analyzed postures and behaviors") 
    print("✅ Policy Engine: Applied rules and detected violations")
    print("✅ Multi-tracking: Monitored multiple students simultaneously")
    print("✅ Real-time Ready: Pipeline optimized for live monitoring")
    
    if any(v.severity == Severity.CHEATING for v in active_violations.values()):
        print("🚨 CHEATING DETECTED - Administrator intervention required!")
    elif active_violations:
        print("⚠️  Suspicious behavior detected - Continue monitoring")
    else:
        print("✅ All students behaving appropriately")
    
    print(f"\n🎓 CheatGPT3 Pipeline Demo Complete!")

if __name__ == "__main__":
    demo_cheatgpt_pipeline()

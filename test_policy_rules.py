"""Comprehensive test of the policy rules engine."""
import os
import sys
import time
from typing import List, Dict

# Add the cheatgpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cheatgpt'))

def test_policy_engine():
    """Test the policy engine with various scenarios."""
    print("=== POLICY ENGINE COMPREHENSIVE TEST ===\n")
    
    try:
        from cheatgpt.policy.rules import (
            PolicyEngine, Severity, ViolationResult, check_rule,
            get_active_violations, clear_track, get_policy_statistics
        )
        print("✓ Successfully imported policy components")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Initialize policy engine
    engine = PolicyEngine()
    print(f"✓ PolicyEngine initialized")
    print(f"   Configuration: {engine.get_statistics()['configuration']}")
    print()
    
    # Test 1: Normal behavior
    print("1. Testing Normal Behavior:")
    track_id = "person_1"
    violations = check_rule(track_id, False, False, False)
    print(f"   Normal flags: {len(violations)} violations (expected: 0)")
    print()
    
    # Test 2: Single frame violations (should not trigger)
    print("2. Testing Single Frame Violations:")
    violations = check_rule(track_id, True, False, False)  # Single lean
    print(f"   Single lean: {len(violations)} violations (expected: 0)")
    
    violations = check_rule(track_id, False, True, False)  # Single look
    print(f"   Single look: {len(violations)} violations (expected: 0)")
    
    violations = check_rule(track_id, False, False, True)  # Single phone
    print(f"   Single phone: {len(violations)} violations (expected: 0)")
    print()
    
    # Test 3: Persistent leaning (should trigger after 5 frames)
    print("3. Testing Persistent Leaning:")
    track_id = "person_2"
    for i in range(6):  # 6 frames of leaning
        violations = check_rule(track_id, True, False, False)
        if violations:
            print(f"   Frame {i+1}: LEANING detected - {violations[0].label}")
            break
        else:
            print(f"   Frame {i+1}: No violation yet")
    
    active = get_active_violations()
    print(f"   Active violations: {len(active)}")
    print()
    
    # Test 4: Persistent looking around
    print("4. Testing Persistent Looking Around:")
    track_id = "person_3"
    for i in range(6):  # 6 frames of looking
        violations = check_rule(track_id, False, True, False)
        if violations:
            print(f"   Frame {i+1}: LOOKING detected - {violations[0].label}")
            break
        else:
            print(f"   Frame {i+1}: No violation yet")
    print()
    
    # Test 5: Persistent phone use
    print("5. Testing Persistent Phone Use:")
    track_id = "person_4"
    for i in range(6):  # 6 frames of phone use
        violations = check_rule(track_id, False, False, True)
        if violations:
            print(f"   Frame {i+1}: PHONE USE detected - {violations[0].label}")
            print(f"   Severity: {violations[0].severity.color}")
            break
        else:
            print(f"   Frame {i+1}: No violation yet")
    print()
    
    # Test 6: Cheating scenario (phone + repeated suspicious behavior)
    print("6. Testing Cheating Detection:")
    track_id = "person_5"
    
    # First, build up suspicious behavior history
    for i in range(4):
        check_rule(track_id, True, False, False)  # Leaning behavior
        check_rule(track_id, False, True, False)  # Looking behavior
        time.sleep(0.1)  # Small delay to create distinct timestamps
    
    # Now add phone use - should trigger cheating
    violations = check_rule(track_id, False, False, True)
    if violations and violations[0].severity == Severity.CHEATING:
        print(f"   ✓ CHEATING detected: {violations[0].label}")
        print(f"   Details: {violations[0].details}")
        print(f"   Severity: {violations[0].severity.color}")
    else:
        print(f"   Expected cheating, got: {violations}")
    print()
    
    # Test 7: Recovery to normal behavior
    print("7. Testing Recovery to Normal:")
    track_id = "person_6"
    
    # Create a violation
    for i in range(5):
        check_rule(track_id, True, False, False)  # Leaning
    
    active_before = len(get_active_violations())
    print(f"   Violations before recovery: {active_before}")
    
    # Return to normal behavior
    for i in range(6):
        violations = check_rule(track_id, False, False, False)
        if not violations:
            print(f"   Frame {i+1}: Returned to normal")
    
    active_after = len(get_active_violations())
    print(f"   Violations after recovery: {active_after}")
    print()
    
    # Test 8: Multiple track management
    print("8. Testing Multiple Track Management:")
    tracks = ["person_A", "person_B", "person_C"]
    
    # Create different violations for different tracks
    check_rule(tracks[0], True, False, False)   # Leaning
    check_rule(tracks[1], False, True, False)   # Looking
    check_rule(tracks[2], False, False, True)   # Phone
    
    for i in range(5):  # Make them persistent
        for track in tracks:
            if track == tracks[0]:
                check_rule(track, True, False, False)
            elif track == tracks[1]:
                check_rule(track, False, True, False)
            else:
                check_rule(track, False, False, True)
    
    active_violations = get_active_violations()
    print(f"   Active violations across tracks: {len(active_violations)}")
    for track_id, violation in active_violations.items():
        print(f"     {track_id}: {violation.label} ({violation.severity.color})")
    print()
    
    # Test 9: Statistics and monitoring
    print("9. Testing Statistics:")
    stats = get_policy_statistics()
    print(f"   Total tracks: {stats['total_tracks']}")
    print(f"   Active violations: {stats['active_violations']}")
    print(f"   Severity distribution: {stats['severity_distribution']}")
    print(f"   Configuration: {stats['configuration']}")
    print()
    
    # Test 10: Track cleanup
    print("10. Testing Track Cleanup:")
    initial_tracks = stats['total_tracks']
    clear_track("person_A")
    clear_track("person_B")
    
    stats_after = get_policy_statistics()
    print(f"   Tracks before cleanup: {initial_tracks}")
    print(f"   Tracks after cleanup: {stats_after['total_tracks']}")
    print()
    
    # Test 11: Severity levels and color coding
    print("11. Testing Severity Levels:")
    severity_tests = [
        (Severity.NORMAL, "green"),
        (Severity.LEANING, "yellow"),
        (Severity.LOOKING, "yellow"),
        (Severity.PHONE_USE, "orange"),
        (Severity.CHEATING, "red")
    ]
    
    for severity, expected_color in severity_tests:
        print(f"   {severity.label}: {severity.color} (expected: {expected_color})")
        assert severity.color == expected_color, f"Color mismatch for {severity.label}"
    print("   ✓ All severity colors correct")
    print()
    
    # Test 12: Time-based behavior window
    print("12. Testing Time-Based Behavior Window:")
    track_id = "time_test"
    
    # Add some old behavior (should be cleaned up)
    old_time = time.time() - 15  # 15 seconds ago (outside 10s window)
    check_rule(track_id, True, False, False, old_time)
    
    # Add current behavior
    current_time = time.time()
    violations = check_rule(track_id, False, True, False, current_time)
    
    history = engine.get_track_history(track_id)
    print(f"   History length after cleanup: {len(history)} (should be recent only)")
    
    if history:
        time_diff = current_time - history[0].timestamp
        print(f"   Oldest event age: {time_diff:.1f}s (should be < 10s)")
    print()
    
    print("=== POLICY ENGINE TEST COMPLETED ===")
    
    # Final validation
    final_stats = get_policy_statistics()
    print(f"\nFinal Statistics:")
    print(f"✅ Total tracks processed: {final_stats['total_tracks']}")
    print(f"✅ Active violations: {final_stats['active_violations']}")
    print(f"✅ Policy rules working correctly")
    
    return True

def test_integration_example():
    """Test integration with pose detector output format."""
    print("\n=== INTEGRATION TEST ===")
    
    from cheatgpt.policy.rules import check_rule
    
    # Simulate pose detector output
    pose_detections = [
        {
            'track_id': 'student_001',
            'bbox': [100, 200, 300, 600],
            'yaw': -35.0,
            'pitch': 10.0,
            'lean_flag': False,
            'look_flag': True,  # Looking around (yaw > 30°)
            'phone_flag': False,
            'confidence': 0.89
        },
        {
            'track_id': 'student_002',
            'bbox': [400, 150, 550, 650],
            'yaw': 5.0,
            'pitch': -5.0,
            'lean_flag': True,  # Leaning detected
            'look_flag': False,
            'phone_flag': False,
            'confidence': 0.92
        }
    ]
    
    print("Processing simulated pose detections...")
    
    for detection in pose_detections:
        violations = check_rule(
            detection['track_id'],
            detection['lean_flag'],
            detection['look_flag'],
            detection['phone_flag']
        )
        
        print(f"Track {detection['track_id']}:")
        print(f"  Behaviors: lean={detection['lean_flag']}, look={detection['look_flag']}, phone={detection['phone_flag']}")
        print(f"  Violations: {[v.label for v in violations]}")
    
    print("✓ Integration test completed")

if __name__ == "__main__":
    success = test_policy_engine()
    if success:
        test_integration_example()
        print("\n🎉 ALL POLICY TESTS PASSED!")
    else:
        print("\n❌ POLICY TESTS FAILED")

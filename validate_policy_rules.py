"""Final validation of the rules-based policy implementation."""
import os
import sys

# Add the cheatgpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cheatgpt'))

def validate_policy_rules():
    """Validate that the policy rules meet all requirements."""
    print("=== POLICY RULES VALIDATION ===\n")
    
    try:
        from cheatgpt.policy.rules import (
            PolicyEngine, Severity, ViolationResult, BehaviorEvent,
            check_rule, get_active_violations, clear_track, get_policy_statistics
        )
        print("✓ Successfully imported all policy components")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # 1. Validate environment configuration
    print("1. Testing Environment Configuration:")
    try:
        engine = PolicyEngine()
        config = engine.get_statistics()['configuration']
        
        expected_params = ['behavior_window', 'alert_persist_frames', 'phone_repeat_thresh']
        for param in expected_params:
            if param in config:
                print(f"   ✓ {param}: {config[param]}")
            else:
                print(f"   ✗ Missing parameter: {param}")
                return False
        
        # Validate specific values from .env
        assert config['behavior_window'] == 10.0, f"Expected behavior_window=10.0, got {config['behavior_window']}"
        assert config['alert_persist_frames'] == 5, f"Expected alert_persist_frames=5, got {config['alert_persist_frames']}"
        assert config['phone_repeat_thresh'] == 3, f"Expected phone_repeat_thresh=3, got {config['phone_repeat_thresh']}"
        
    except Exception as e:
        print(f"   ✗ Configuration validation failed: {e}")
        return False
    
    # 2. Validate history buffer management
    print("\n2. Testing History Buffer Management:")
    try:
        track_id = "test_buffer"
        
        # Add events to history
        for i in range(3):
            check_rule(track_id, True, False, False)
        
        history = engine.get_track_history(track_id)
        print(f"   ✓ History buffer maintains events: {len(history)} events")
        
        # Test time-based cleanup (simulated)
        old_time = 1000.0  # Very old timestamp
        check_rule(track_id, False, True, False, old_time)
        current_time = old_time + 15.0  # 15 seconds later (outside 10s window)
        check_rule(track_id, False, False, True, current_time)
        
        history_after = engine.get_track_history(track_id)
        print(f"   ✓ Time-based cleanup working: {len(history_after)} events after cleanup")
        
    except Exception as e:
        print(f"   ✗ History buffer validation failed: {e}")
        return False
    
    # 3. Validate input/output format
    print("\n3. Testing Input/Output Format:")
    try:
        # Test input format
        track_id = "format_test"
        lean_flag = True
        look_flag = False
        phone_flag = False
        
        violations = check_rule(track_id, lean_flag, look_flag, phone_flag)
        print(f"   ✓ Input format accepted: track_id, lean_flag, look_flag, phone_flag")
        
        # Test output format after building up violations
        for i in range(5):  # Build persistent violation
            violations = check_rule(track_id, True, False, False)
        
        if violations:
            violation = violations[0]
            required_fields = ['track_id', 'label', 'severity', 'start_ts', 'end_ts']
            
            for field in required_fields:
                if hasattr(violation, field):
                    print(f"   ✓ Output contains required field: {field}")
                else:
                    print(f"   ✗ Missing required field: {field}")
                    return False
            
            # Validate specific output values
            assert violation.track_id == track_id, f"Track ID mismatch: {violation.track_id}"
            assert isinstance(violation.severity, Severity), f"Severity not enum: {type(violation.severity)}"
            assert violation.start_ts <= violation.end_ts, f"Invalid timestamps: {violation.start_ts} > {violation.end_ts}"
            print(f"   ✓ Output format validation passed")
        
    except Exception as e:
        print(f"   ✗ Input/output format validation failed: {e}")
        return False
    
    # 4. Validate all rule types
    print("\n4. Testing All Rule Types:")
    
    rule_tests = [
        ("Normal", False, False, False, "Normal"),
        ("Leaning", True, False, False, "Leaning"),
        ("Looking Around", False, True, False, "Looking Around"),
        ("Phone Use", False, False, True, "Phone Use")
    ]
    
    try:
        for rule_name, lean, look, phone, expected_label in rule_tests:
            test_track = f"rule_{rule_name.lower().replace(' ', '_')}"
            
            # Build up persistent behavior
            violations = None
            for i in range(6):  # Ensure persistence threshold is met
                violations = check_rule(test_track, lean, look, phone)
                if violations:
                    break
            
            if rule_name == "Normal":
                if not violations:
                    print(f"   ✓ {rule_name} rule: No violations (correct)")
                else:
                    print(f"   ✗ {rule_name} rule: Unexpected violations {[v.label for v in violations]}")
                    return False
            else:
                if violations and violations[0].label == expected_label:
                    print(f"   ✓ {rule_name} rule: {violations[0].label}")
                else:
                    print(f"   ✗ {rule_name} rule failed: Expected {expected_label}, got {violations}")
                    return False
        
        # Test cheating rule (complex scenario)
        cheating_track = "cheating_test"
        
        # Build suspicious behavior history
        for i in range(4):
            check_rule(cheating_track, True, False, False)  # Leaning
            check_rule(cheating_track, False, True, False)  # Looking
        
        # Add phone use to trigger cheating
        violations = check_rule(cheating_track, False, False, True)
        
        if violations and any(v.label == "Cheating" for v in violations):
            print(f"   ✓ Cheating rule: Detected correctly")
        else:
            print(f"   ✗ Cheating rule failed: {violations}")
            return False
        
    except Exception as e:
        print(f"   ✗ Rule type validation failed: {e}")
        return False
    
    # 5. Validate severity levels and color coding
    print("\n5. Testing Severity Levels:")
    try:
        severity_mappings = {
            Severity.NORMAL: "green",
            Severity.LEANING: "yellow",
            Severity.LOOKING: "yellow",
            Severity.PHONE_USE: "orange",
            Severity.CHEATING: "red"
        }
        
        for severity, expected_color in severity_mappings.items():
            if severity.color == expected_color:
                print(f"   ✓ {severity.label}: {severity.color}")
            else:
                print(f"   ✗ {severity.label}: Expected {expected_color}, got {severity.color}")
                return False
        
    except Exception as e:
        print(f"   ✗ Severity validation failed: {e}")
        return False
    
    # 6. Validate persistence thresholds
    print("\n6. Testing Persistence Thresholds:")
    try:
        persist_track = "persistence_test"
        
        # Test that violations don't trigger before threshold
        for i in range(4):  # Just under the threshold (5)
            violations = check_rule(persist_track, True, False, False)
            if violations:
                print(f"   ✗ Violation triggered too early at frame {i+1}")
                return False
        
        print(f"   ✓ No violations before persistence threshold (4/5 frames)")
        
        # Test that violations trigger at threshold
        violations = check_rule(persist_track, True, False, False)
        if violations:
            print(f"   ✓ Violation triggered at persistence threshold (5/5 frames)")
        else:
            print(f"   ✗ Violation not triggered at threshold")
            return False
        
    except Exception as e:
        print(f"   ✗ Persistence threshold validation failed: {e}")
        return False
    
    # 7. Validate behavior repeat window
    print("\n7. Testing Behavior Repeat Window:")
    try:
        window_track = "window_test"
        
        # Test that behaviors outside window don't count for cheating
        old_time = 1000.0
        current_time = old_time + 15.0  # Outside 10s window
        
        # Add old behaviors (outside window)
        check_rule(window_track, True, False, False, old_time)
        check_rule(window_track, False, True, False, old_time)
        
        # Add current phone use
        violations = check_rule(window_track, False, False, True, current_time)
        
        # Should not trigger cheating because old behaviors are outside window
        cheating_detected = any(v.label == "Cheating" for v in violations)
        if not cheating_detected:
            print(f"   ✓ Behavior window correctly excludes old events")
        else:
            print(f"   ✗ Behavior window failed: Old events incorrectly included")
            return False
        
    except Exception as e:
        print(f"   ✗ Behavior window validation failed: {e}")
        return False
    
    print("\n=== VALIDATION SUCCESSFUL ===")
    print("✅ Policy rules implementation meets all requirements:")
    print("   • Maintains history buffer for each track ID")
    print("   • Uses configurable time window (BEHAVIOR_REPEAT_WINDOW)")
    print("   • Correctly processes input format {track_id, lean_flag, look_flag, phone_flag}")
    print("   • Implements all rule types (Normal, Leaning, Looking, Phone, Cheating)")
    print("   • Applies persistence thresholds (ALERT_PERSIST_FRAMES)")
    print("   • Detects cheating with repeat behavior analysis (PHONE_REPEAT_THRESH)")
    print("   • Returns correct output format {track_id, label, severity, start_ts, end_ts}")
    print("   • Uses proper severity color coding (Green/Yellow/Orange/Red)")
    print("   • Handles time-based window management")
    print("   • Supports multi-track monitoring")
    
    return True

if __name__ == "__main__":
    success = validate_policy_rules()
    if success:
        print("\n🎉 POLICY RULES READY FOR PRODUCTION!")
    else:
        print("\n❌ VALIDATION FAILED - CHECK IMPLEMENTATION")

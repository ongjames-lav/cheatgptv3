"""
Quick test script to validate optimized rule system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cheatgpt.policy.rules import PolicyEngine, Severity
import time

def test_optimized_rules():
    """Test the optimized rule system."""
    print("🧪 Testing Optimized CheatGPT3 Rule System")
    print("=" * 50)
    
    # Initialize policy engine
    engine = PolicyEngine()
    print(f"✅ Policy engine initialized with settings:")
    stats = engine.get_statistics()
    config = stats['configuration']
    print(f"   - Behavior window: {config['behavior_window']}s")
    print(f"   - Alert persist frames: {config['alert_persist_frames']}")  
    print(f"   - Phone repeat threshold: {config['phone_repeat_thresh']}")
    print(f"   - Debug mode: {engine.debug_mode}")
    print()
    
    track_id = "test_person_001"
    
    # Test 1: Normal behavior
    print("🟢 Test 1: Normal behavior")
    violations = engine.update_behavior(track_id, False, False, False)
    print(f"   Violations detected: {len(violations)}")
    print()
    
    # Test 2: Leaning behavior
    print("🟡 Test 2: Leaning behavior")
    violations = engine.update_behavior(track_id, True, False, False)
    print(f"   Violations detected: {len(violations)}")
    if violations:
        print(f"   Severity: {violations[0].severity.label}")
        print(f"   Details: {violations[0].details}")
    print()
    
    # Test 3: Looking around
    print("🟡 Test 3: Looking around")  
    violations = engine.update_behavior(track_id, False, True, False)
    print(f"   Violations detected: {len(violations)}")
    if violations:
        print(f"   Severity: {violations[0].severity.label}")
        print(f"   Details: {violations[0].details}")
    print()
    
    # Test 4: Phone use
    print("🟠 Test 4: Phone use")
    violations = engine.update_behavior(track_id, False, False, True)
    print(f"   Violations detected: {len(violations)}")
    if violations:
        print(f"   Severity: {violations[0].severity.label}")
        print(f"   Details: {violations[0].details}")
    print()
    
    # Test 5: Cheating behavior (phone + suspicious)
    print("🔴 Test 5: Cheating behavior (phone + leaning)")
    violations = engine.update_behavior(track_id, True, False, True)
    print(f"   Violations detected: {len(violations)}")
    if violations:
        print(f"   Severity: {violations[0].severity.label}")
        print(f"   Details: {violations[0].details}")
    print()
    
    # Test 6: Multiple suspicious behaviors with phone
    print("🔴 Test 6: Multiple suspicious behaviors + phone")
    violations = engine.update_behavior(track_id, True, True, True)
    print(f"   Violations detected: {len(violations)}")
    if violations:
        print(f"   Severity: {violations[0].severity.label}")
        print(f"   Details: {violations[0].details}")
    print()
    
    # Test 7: Active violations
    active = engine.get_active_violations()
    print(f"📊 Active violations: {len(active)}")
    for track, violation in active.items():
        print(f"   {track}: {violation.severity.label} - {violation.details}")
    print()
    
    # Test 8: Return to normal
    print("🟢 Test 8: Return to normal behavior")
    violations = engine.update_behavior(track_id, False, False, False)
    print(f"   Violations detected: {len(violations)}")
    
    # Check if violation was cleared
    time.sleep(0.1)  # Small delay
    violations = engine.update_behavior(track_id, False, False, False)
    active = engine.get_active_violations()
    print(f"   Active violations after normal behavior: {len(active)}")
    print()
    
    print("✅ Rule system test complete!")
    print(f"📈 Final statistics: {engine.get_statistics()}")

if __name__ == "__main__":
    test_optimized_rules()

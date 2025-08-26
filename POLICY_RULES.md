# Rules-Based Policy Implementation

## Overview
The rules-based policy engine implements sophisticated behavioral analysis for the CheatGPT3 exam proctoring system. It maintains behavioral history for each tracked person and applies configurable rules to detect violations with different severity levels.

## Implementation Details

### Files
- `cheatgpt/policy/rules.py` - Main policy engine implementation
- `cheatgpt/.env` - Configuration (updated with policy settings)

### Key Components

#### 1. PolicyEngine Class
Core engine that manages behavior tracking and rule evaluation:
- **History Management**: Maintains time-windowed behavior history per track
- **Rule Evaluation**: Applies persistence-based violation detection
- **Multi-track Support**: Concurrent monitoring of multiple people
- **Time-based Cleanup**: Automatic removal of expired behavior events

#### 2. Severity Levels
Color-coded violation severity system:
```python
class Severity(Enum):
    NORMAL = ("Normal", "green")           # No violations
    LEANING = ("Leaning", "yellow")        # Minor postural violation
    LOOKING = ("Looking Around", "yellow")  # Head movement violation
    PHONE_USE = ("Phone Use", "orange")    # Device usage violation
    CHEATING = ("Cheating", "red")         # Major violation (escalated)
```

#### 3. Behavior Tracking
Time-stamped behavior events with automatic window management:
```python
@dataclass
class BehaviorEvent:
    timestamp: float
    lean_flag: bool
    look_flag: bool
    phone_flag: bool
```

### Rule Logic

#### Input Format
```python
{
    'track_id': str,      # Person identifier
    'lean_flag': bool,    # True if leaning detected
    'look_flag': bool,    # True if looking around
    'phone_flag': bool    # True if phone near person
}
```

#### Rule Evaluation Hierarchy

1. **Normal Behavior**
   - Condition: No flags active
   - Action: No violation

2. **Leaning Detection**
   - Condition: `lean_flag == True` for ≥ `ALERT_PERSIST_FRAMES`
   - Severity: Yellow
   - Threshold: 5 consecutive frames (default)

3. **Looking Around Detection**
   - Condition: `look_flag == True` for ≥ `ALERT_PERSIST_FRAMES`
   - Severity: Yellow
   - Threshold: 5 consecutive frames (default)

4. **Phone Use Detection**
   - Condition: `phone_flag == True` for ≥ `ALERT_PERSIST_FRAMES`
   - Severity: Orange
   - Threshold: 5 consecutive frames (default)

5. **Cheating Detection** (Highest Priority)
   - Condition: `phone_flag == True` AND (`lean_flag` OR `look_flag`) repeats ≥ `PHONE_REPEAT_THRESH` in last `BEHAVIOR_REPEAT_WINDOW` seconds
   - Severity: Red
   - Complex pattern analysis combining phone use with suspicious behaviors

#### Output Format
```python
{
    'track_id': str,        # Person identifier
    'label': str,           # Violation type name
    'severity': Severity,   # Severity enum with color
    'start_ts': float,      # Violation start timestamp
    'end_ts': float,        # Violation end timestamp
    'confidence': float,    # Detection confidence (0-1)
    'details': str          # Detailed description
}
```

### Configuration

Environment variables in `.env`:
```bash
# Policy Rules Configuration
BEHAVIOR_REPEAT_WINDOW=10.0    # Time window for behavior analysis (seconds)
ALERT_PERSIST_FRAMES=5         # Frames required for persistent violation
PHONE_REPEAT_THRESH=3          # Suspicious behaviors needed for cheating detection
```

### Usage Examples

#### Basic Usage
```python
from cheatgpt.policy.rules import check_rule, get_active_violations

# Process behavior observation
violations = check_rule(
    track_id="student_001",
    lean_flag=True,
    look_flag=False,
    phone_flag=False
)

# Check current violations
active = get_active_violations()
```

#### Integration with Pose Detection
```python
# Get pose data
poses = pose_detector.estimate(frame, phone_detections)

# Apply policy rules
for pose in poses:
    violations = check_rule(
        pose['track_id'],
        pose['lean_flag'],
        pose['look_flag'],
        pose['phone_flag']
    )
    
    for violation in violations:
        if violation.severity == Severity.CHEATING:
            alert_administrator(violation)
```

#### Multi-track Monitoring
```python
# Process multiple students
for student_data in current_detections:
    violations = check_rule(
        student_data['track_id'],
        student_data['lean_flag'],
        student_data['look_flag'],
        student_data['phone_flag']
    )

# Get comprehensive status
stats = get_policy_statistics()
print(f"Monitoring {stats['total_tracks']} students")
print(f"Active violations: {stats['active_violations']}")
```

### Advanced Features

#### 1. Temporal Analysis
- **Time Windows**: Configurable behavior analysis windows
- **Event Cleanup**: Automatic removal of expired events
- **Trend Detection**: Pattern analysis over time

#### 2. Escalation Logic
- **Progressive Severity**: Minor violations can escalate to major ones
- **Context Awareness**: Phone use combined with suspicious behaviors triggers cheating
- **Confidence Scoring**: Different confidence levels per violation type

#### 3. State Management
- **Persistent Tracking**: Maintains state across video frames
- **Recovery Detection**: Automatic violation clearing when behavior normalizes
- **Memory Management**: Efficient cleanup of inactive tracks

### Testing and Validation

Comprehensive test suite validates:
- ✅ **Configuration Loading**: Environment variable integration
- ✅ **Rule Logic**: All violation types and thresholds
- ✅ **Persistence**: Frame-based persistence requirements
- ✅ **Escalation**: Cheating detection with complex patterns
- ✅ **Time Management**: Window-based behavior analysis
- ✅ **Multi-tracking**: Concurrent student monitoring
- ✅ **Integration**: Compatibility with pose detection pipeline

### Performance Characteristics

- **Memory Efficient**: Automatic cleanup of old events
- **Real-time Ready**: Optimized for video frame processing
- **Scalable**: Supports monitoring multiple students simultaneously
- **Configurable**: Adjustable thresholds without code changes

### Integration Points

The policy engine integrates with:
- **Pose Detector**: Receives behavioral flags from pose analysis
- **Tracking System**: Uses track IDs for person identification
- **Alert System**: Triggers notifications for violations
- **Database**: Stores violation events and statistics
- **Reporting**: Provides data for comprehensive reports

### Validation Results

✅ **All Requirements Met**:
- History buffer management with configurable time windows
- Input format: `{track_id, lean_flag, look_flag, phone_flag}`
- All rule types implemented with correct logic
- Persistence thresholds properly enforced
- Complex cheating detection with repeat behavior analysis
- Output format: `{track_id, label, severity, start_ts, end_ts}`
- Color-coded severity levels (Green/Yellow/Orange/Red)
- Multi-track support with efficient state management

The rules-based policy engine is fully operational and ready for production deployment in the CheatGPT3 exam proctoring system.

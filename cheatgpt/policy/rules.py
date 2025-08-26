"""Rules-based policy engine for detecting cheating behaviors and violations.

This module implements a sophisticated policy system that maintains behavior history
for each tracked person and applies rules to detect various levels of violations.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Severity(Enum):
    """Violation severity levels with color coding."""
    NORMAL = ("Normal", "green")
    LEANING = ("Leaning", "yellow")
    LOOKING = ("Looking Around", "yellow")
    PHONE_USE = ("Phone Use", "orange")
    CHEATING = ("Cheating", "red")
    
    def __init__(self, label: str, color: str):
        self.label = label
        self.color = color

@dataclass
class BehaviorEvent:
    """Represents a single behavior observation."""
    timestamp: float
    lean_flag: bool
    look_flag: bool
    phone_flag: bool

@dataclass
class ViolationResult:
    """Result of policy evaluation."""
    track_id: str
    label: str
    severity: Severity
    start_ts: float
    end_ts: float
    confidence: float = 1.0
    details: str = ""

class PolicyEngine:
    """Rules-based policy engine for behavioral analysis."""
    
    def __init__(self):
        """Initialize the policy engine with configuration from environment."""
        # Load configuration from environment with better defaults
        self.behavior_window = float(os.getenv('BEHAVIOR_REPEAT_WINDOW', '5.0'))  # Reduced for faster response
        self.alert_persist_frames = int(os.getenv('ALERT_PERSIST_FRAMES', '3'))  # Reduced for more sensitivity
        self.phone_repeat_thresh = int(os.getenv('PHONE_REPEAT_THRESH', '2'))    # Reduced for more sensitivity
        
        # Additional thresholds for better detection
        self.lean_threshold = float(os.getenv('LEAN_THRESHOLD', '0.3'))
        self.look_threshold = float(os.getenv('LOOK_THRESHOLD', '15.0'))  # degrees
        self.phone_iou_threshold = float(os.getenv('PHONE_IOU_THRESHOLD', '0.1'))
        
        # History buffers for each track ID
        self.behavior_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.active_violations: Dict[str, ViolationResult] = {}
        self.violation_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Debug tracking
        self.debug_mode = os.getenv('DEBUG_POLICY', 'false').lower() == 'true'
        
        logger.info(f"PolicyEngine initialized with window={self.behavior_window}s, "
                   f"persist_frames={self.alert_persist_frames}, "
                   f"phone_thresh={self.phone_repeat_thresh}, debug={self.debug_mode}")
    
    def update_behavior(self, track_id: str, lean_flag: bool, look_flag: bool, 
                       phone_flag: bool, timestamp: Optional[float] = None) -> List[ViolationResult]:
        """
        Update behavior history for a track and evaluate policy rules.
        
        Args:
            track_id: Unique identifier for the tracked person
            lean_flag: True if person is leaning
            look_flag: True if person is looking around
            phone_flag: True if phone is near person
            timestamp: Event timestamp (uses current time if None)
            
        Returns:
            List of current violations for this track
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Debug logging
        if self.debug_mode:
            logger.info(f"Policy update for {track_id}: lean={lean_flag}, look={look_flag}, phone={phone_flag}")
        
        # Create behavior event
        event = BehaviorEvent(timestamp, lean_flag, look_flag, phone_flag)
        
        # Add to history
        self.behavior_history[track_id].append(event)
        
        # Clean old events outside the window
        self._clean_history(track_id, timestamp)
        
        # Evaluate rules
        violations = self._evaluate_rules(track_id, timestamp)
        
        # Update active violations
        if violations:
            self.active_violations[track_id] = violations[0]  # Use highest severity
            if self.debug_mode:
                logger.info(f"Active violation for {track_id}: {violations[0].severity.label}")
        elif track_id in self.active_violations:
            # Check if violation should end
            if self._should_end_violation(track_id, timestamp):
                if self.debug_mode:
                    logger.info(f"Ending violation for {track_id}")
                del self.active_violations[track_id]
        
        return violations
    
    def _clean_history(self, track_id: str, current_timestamp: float):
        """Remove events outside the behavior window."""
        history = self.behavior_history[track_id]
        cutoff_time = current_timestamp - self.behavior_window
        
        while history and history[0].timestamp < cutoff_time:
            history.popleft()
    
    def _evaluate_rules(self, track_id: str, timestamp: float) -> List[ViolationResult]:
        """Evaluate all policy rules for a track."""
        history = self.behavior_history[track_id]
        violations = []
        
        if not history:
            return violations
        
        # Get recent events for immediate response
        recent_events = list(history)[-self.alert_persist_frames:] if len(history) >= self.alert_persist_frames else list(history)
        current_event = history[-1]
        
        # Count recent behaviors (for immediate alerts)
        recent_lean_count = sum(1 for event in recent_events if event.lean_flag)
        recent_look_count = sum(1 for event in recent_events if event.look_flag)
        recent_phone_count = sum(1 for event in recent_events if event.phone_flag)
        
        # Count behaviors in full window for pattern analysis
        window_lean_count = sum(1 for event in history if event.lean_flag)
        window_look_count = sum(1 for event in history if event.look_flag)
        window_phone_count = sum(1 for event in history if event.phone_flag)
        
        if self.debug_mode:
            logger.info(f"Rule evaluation for {track_id}: recent_lean={recent_lean_count}, "
                       f"recent_look={recent_look_count}, recent_phone={recent_phone_count}, "
                       f"window_lean={window_lean_count}, window_look={window_look_count}, "
                       f"window_phone={window_phone_count}")
        
        # Rule 1: Immediate Cheating Detection (highest priority)
        if current_event.phone_flag and (current_event.lean_flag or current_event.look_flag):
            violation = ViolationResult(
                track_id=track_id,
                label=Severity.CHEATING.label,
                severity=Severity.CHEATING,
                start_ts=self._get_violation_start_time(track_id, history, 'cheating'),
                end_ts=timestamp,
                confidence=0.98,
                details=f"Phone use with suspicious behavior detected"
            )
            violations.append(violation)
            if self.debug_mode:
                logger.warning(f"IMMEDIATE CHEATING detected for {track_id}")
        
        # Rule 2: Pattern-based Cheating Detection
        elif self._check_cheating_rule(track_id, history, timestamp):
            violation = ViolationResult(
                track_id=track_id,
                label=Severity.CHEATING.label,
                severity=Severity.CHEATING,
                start_ts=self._get_violation_start_time(track_id, history, 'cheating'),
                end_ts=timestamp,
                confidence=0.95,
                details=f"Phone use with repeated suspicious behavior (phone:{window_phone_count}, "
                       f"lean:{window_lean_count}, look:{window_look_count})"
            )
            violations.append(violation)
            if self.debug_mode:
                logger.warning(f"PATTERN CHEATING detected for {track_id}: {violation.details}")
        
        # Rule 3: Persistent Phone Use
        elif recent_phone_count >= min(self.alert_persist_frames, 2) or current_event.phone_flag:
            violation = ViolationResult(
                track_id=track_id,
                label=Severity.PHONE_USE.label,
                severity=Severity.PHONE_USE,
                start_ts=self._get_violation_start_time(track_id, history, 'phone'),
                end_ts=timestamp,
                confidence=0.85,
                details=f"Phone detected ({recent_phone_count} recent frames)"
            )
            violations.append(violation)
            if self.debug_mode:
                logger.warning(f"PHONE USE detected for {track_id}")
        
        # Rule 4: Persistent Looking Around
        elif recent_look_count >= min(self.alert_persist_frames, 2) or (current_event.look_flag and recent_look_count >= 1):
            violation = ViolationResult(
                track_id=track_id,
                label=Severity.LOOKING.label,
                severity=Severity.LOOKING,
                start_ts=self._get_violation_start_time(track_id, history, 'look'),
                end_ts=timestamp,
                confidence=0.75,
                details=f"Looking around detected ({recent_look_count} recent frames)"
            )
            violations.append(violation)
            if self.debug_mode:
                logger.info(f"LOOKING AROUND detected for {track_id}")
        
        # Rule 5: Persistent Leaning
        elif recent_lean_count >= min(self.alert_persist_frames, 2) or (current_event.lean_flag and recent_lean_count >= 1):
            violation = ViolationResult(
                track_id=track_id,
                label=Severity.LEANING.label,
                severity=Severity.LEANING,
                start_ts=self._get_violation_start_time(track_id, history, 'lean'),
                end_ts=timestamp,
                confidence=0.70,
                details=f"Leaning detected ({recent_lean_count} recent frames)"
            )
            violations.append(violation)
            if self.debug_mode:
                logger.info(f"LEANING detected for {track_id}")
        
        # Rule 6: Normal behavior (no violations)
        else:
            # Only log if there was a previous violation
            if track_id in self.active_violations and self.debug_mode:
                logger.info(f"NORMAL behavior resumed for {track_id}")
        
        return violations
    
    def _check_cheating_rule(self, track_id: str, history: deque, timestamp: float) -> bool:
        """
        Check if cheating behavior is detected.
        
        Cheating occurs when:
        - phone_flag == True AND
        - (lean_flag OR look_flag) repeats >= PHONE_REPEAT_THRESH in the last window
        """
        if not history:
            return False
        
        current_event = history[-1]
        
        # Must have current phone usage
        if not current_event.phone_flag:
            return False
        
        # Count lean or look behaviors in the window
        suspicious_behavior_count = 0
        for event in history:
            if event.lean_flag or event.look_flag:
                suspicious_behavior_count += 1
        
        # Check if threshold is met
        return suspicious_behavior_count >= self.phone_repeat_thresh
    
    def _get_violation_start_time(self, track_id: str, history: deque, 
                                 violation_type: str) -> float:
        """Get the start timestamp of a violation based on when it first occurred."""
        if not history:
            return time.time()
        
        # Look for the first occurrence of this violation type in recent frames
        target_frames = min(self.alert_persist_frames, len(history))
        
        for i in range(len(history) - target_frames, len(history)):
            if i < 0:
                continue
            
            event = history[i]
            
            if violation_type == 'lean' and event.lean_flag:
                return event.timestamp
            elif violation_type == 'look' and event.look_flag:
                return event.timestamp
            elif violation_type == 'phone' and event.phone_flag:
                return event.timestamp
            elif violation_type == 'cheating' and (event.phone_flag or event.lean_flag or event.look_flag):
                return event.timestamp
        
        # Fallback to most recent event
        return history[-1].timestamp
    
    def _should_end_violation(self, track_id: str, timestamp: float) -> bool:
        """Check if an active violation should be ended."""
        history = self.behavior_history[track_id]
        
        if not history:
            return True
        
        # Check recent frames for any violations
        recent_events = list(history)[-self.alert_persist_frames:]
        
        has_recent_violation = any(
            event.lean_flag or event.look_flag or event.phone_flag
            for event in recent_events
        )
        
        return not has_recent_violation
    
    def get_active_violations(self) -> Dict[str, ViolationResult]:
        """Get all currently active violations."""
        return self.active_violations.copy()
    
    def get_track_history(self, track_id: str) -> List[BehaviorEvent]:
        """Get behavior history for a specific track."""
        return list(self.behavior_history.get(track_id, []))
    
    def clear_track_history(self, track_id: str):
        """Clear history for a specific track (e.g., when person leaves)."""
        if track_id in self.behavior_history:
            del self.behavior_history[track_id]
        if track_id in self.active_violations:
            del self.active_violations[track_id]
        if track_id in self.violation_counts:
            del self.violation_counts[track_id]
        logger.info(f"Cleared history for track {track_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        total_tracks = len(self.behavior_history)
        active_violations_count = len(self.active_violations)
        
        severity_counts = defaultdict(int)
        for violation in self.active_violations.values():
            severity_counts[violation.severity.label] += 1
        
        return {
            'total_tracks': total_tracks,
            'active_violations': active_violations_count,
            'severity_distribution': dict(severity_counts),
            'configuration': {
                'behavior_window': self.behavior_window,
                'alert_persist_frames': self.alert_persist_frames,
                'phone_repeat_thresh': self.phone_repeat_thresh
            }
        }

# Global policy engine instance
_policy_engine = None

def get_policy_engine() -> PolicyEngine:
    """Get the global policy engine instance."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine()
    return _policy_engine

def check_rule(track_id: str, lean_flag: bool, look_flag: bool, 
               phone_flag: bool, timestamp: Optional[float] = None) -> List[ViolationResult]:
    """
    Check policy rules for a behavior observation.
    
    Args:
        track_id: Unique identifier for the tracked person
        lean_flag: True if person is leaning
        look_flag: True if person is looking around
        phone_flag: True if phone is near person
        timestamp: Event timestamp (uses current time if None)
        
    Returns:
        List of current violations for this track
    """
    engine = get_policy_engine()
    return engine.update_behavior(track_id, lean_flag, look_flag, phone_flag, timestamp)

def get_active_violations() -> Dict[str, ViolationResult]:
    """Get all currently active violations."""
    engine = get_policy_engine()
    return engine.get_active_violations()

def clear_track(track_id: str):
    """Clear history for a specific track."""
    engine = get_policy_engine()
    engine.clear_track_history(track_id)

def get_policy_statistics() -> Dict[str, Any]:
    """Get policy engine statistics."""
    engine = get_policy_engine()
    return engine.get_statistics()

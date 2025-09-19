"""Research-Based Real-time CheatGPT Engine with 30 FPS Live Stream + 10 FPS Detection.

This engine implements classroom cheating detection based on real research with:
- 30 FPS smooth live stream with overlays
- 10 FPS object detection pipeline (every 3rd frame)
- SORT/ByteTrack tracker for frame interpolation
- Research-grounded rule-based temporal smoothing
- Independent 30 FPS recording system
- No LSTM - pure rule-based detection with debouncing

Key Features:
- Live stream never stalls on detection (consistent 30 FPS)
- Detection latency: 50-70ms target
- Temporal smoothing with 30-frame sliding window
- 2-confirmation requirement for cheating events
- Classroom behavior thresholds based on research
"""

import os
import time
import logging
import uuid
from typing import Tuple, List, Dict, Any, Optional, Deque
from collections import deque
import cv2
import numpy as np
import torch
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from ..detectors.yolo11_detector import YOLO11Detector
from ..detectors.pose_detector import PoseDetector
from ..db.db_manager import DBManager
from ..video_recorder import VideoRecorder

# Simple SORT tracker implementation
class SimpleTracker:
    """Simplified object tracker for maintaining detections between frames.
    
    Optimized for classroom environment with up to 40 people.
    """
    
    def __init__(self, max_disappeared: int = 30, max_objects: int = 50):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_objects = max_objects  # Maximum number of objects to track
        self.assignment_threshold = 250  # Increased from 150 to 250 for better single-person tracking
    
    def register(self, bbox: List[float]) -> int:
        """Register a new object."""
        self.objects[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id: int):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update tracker with new detections for classroom environment."""
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            tracked_objects = []
            for detection in detections:
                if len(self.objects) < self.max_objects:
                    object_id = self.register(detection['bbox'])
                    tracked_obj = detection.copy()
                    tracked_obj['track_id'] = object_id
                    tracked_objects.append(tracked_obj)
            return tracked_objects
        
        # Use Hungarian algorithm approach for optimal assignment
        object_ids = list(self.objects.keys())
        object_centroids = [self._get_centroid(self.objects[obj_id]) for obj_id in object_ids]
        detection_centroids = [self._get_centroid(det['bbox']) for det in detections]
        
        # Compute distance matrix
        distances = np.zeros((len(object_centroids), len(detection_centroids)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, det_centroid in enumerate(detection_centroids):
                distances[i, j] = np.linalg.norm(np.array(obj_centroid) - np.array(det_centroid))
        
        # Simple greedy assignment (can be improved with Hungarian algorithm)
        tracked_objects = []
        used_detections = set()
        used_objects = set()
        
        # Sort assignments by distance
        assignments = []
        for i in range(len(object_centroids)):
            for j in range(len(detection_centroids)):
                if distances[i, j] < self.assignment_threshold:
                    assignments.append((distances[i, j], i, j))
        
        assignments.sort(key=lambda x: x[0])  # Sort by distance
        
        # Make assignments
        for distance, obj_idx, det_idx in assignments:
            if obj_idx not in used_objects and det_idx not in used_detections:
                object_id = object_ids[obj_idx]
                self.objects[object_id] = detections[det_idx]['bbox']
                self.disappeared[object_id] = 0
                
                tracked_obj = detections[det_idx].copy()
                tracked_obj['track_id'] = object_id
                tracked_objects.append(tracked_obj)
                
                used_detections.add(det_idx)
                used_objects.add(obj_idx)
        
        # Register unmatched detections as new objects (more conservative for single-person)
        for j, detection in enumerate(detections):
            if j not in used_detections and len(self.objects) < self.max_objects:
                # More conservative registration - only if no existing objects or very far from existing ones
                should_register = True
                if len(self.objects) > 0:
                    # Check if this detection is far enough from existing objects
                    det_centroid = self._get_centroid(detection['bbox'])
                    min_distance = float('inf')
                    for obj_id in self.objects:
                        obj_centroid = self._get_centroid(self.objects[obj_id])
                        distance = np.linalg.norm(np.array(det_centroid) - np.array(obj_centroid))
                        min_distance = min(min_distance, distance)
                    
                    # Only register if far enough from existing objects (avoid duplicate IDs)
                    if min_distance < self.assignment_threshold * 1.5:  # 375 pixels
                        should_register = False
                
                if should_register:
                    object_id = self.register(detection['bbox'])
                    tracked_obj = detection.copy()
                    tracked_obj['track_id'] = object_id
                    tracked_objects.append(tracked_obj)
        
        # Mark unmatched objects as disappeared
        for i, object_id in enumerate(object_ids):
            if i not in used_objects:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return tracked_objects
    
    def _get_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """Get centroid of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class ResearchBasedRuleEngine:
    """Research-based rule engine for classroom cheating detection."""
    
    def __init__(self):
        """Initialize the research-based rule engine."""
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Sliding window configuration (40 frames ≈ 4s at 10 FPS detection - balanced timing)
        self.window_size = 40
        self.detection_fps = 10.0
        self.confirmation_threshold = 3  # Reduced from 6 to 3 for faster detection
        self.phone_confirmation_threshold = 1  # Instant phone detection - no temporal smoothing
        self.hand_confirmation_threshold = 2  # Reduced from 5 to 2 for more sensitive hand detection
        
        # Research-based thresholds for classroom behaviors (realistic settings)
        self.thresholds = {
            # Phone Usage: Detected phone in ≥8 consecutive frames (more accurate)
            'phone_consecutive_frames': 8,
            'phone_duration_threshold': 1.5,  # seconds - longer duration requirement
            
            # Looking Away Frequently: Head yaw >25° left/right, occurring ≥6 times in 4s or held >4s
            'head_turn_angle_threshold': 15.0,  # degrees - more sensitive for smaller movements (was 25.0)
            'head_turn_frequency_threshold': 3,  # occurrences in window - lower threshold (was 6)
            'head_turn_sustained_threshold': 2.0,  # seconds - shorter sustained threshold (was 4.0)
            
            # Looking Down Abnormally: Pitch >35° sustained ≥20 frames (~2.0s)
            'head_pitch_threshold': 20.0,  # degrees - more sensitive threshold (was 35.0)
            'head_pitch_frames_threshold': 10,  # frames - shorter duration (was 20)
            
            # Hand Extended: ≥10 frames (~1.0s) - higher temporal smoothing to reduce sensitivity
            'hand_extended_frames_threshold': 10,  # frames - less sensitive for hand detection
            
            # Out of Frame / Hiding: ≥15 consecutive frames
            'out_of_frame_threshold': 15,  # frames - more realistic threshold
            
            # Normal behavior tolerances (not flagged)
            'normal_head_tilt_threshold': 12.0,  # degrees - tighter tolerance (was 20.0)
            'normal_look_down_duration': 4.0,  # seconds - more generous tolerance
            'normal_hand_adjustment_duration': 5.0,  # seconds - more generous tolerance
            
            # Debounce logic: reset counters if normal posture resumes for ≥3s
            'normal_posture_reset_duration': 3.0  # seconds - faster reset time for responsiveness
        }
        
        # Sliding window tracking per person (classroom environment)
        self.person_windows: Dict[int, Dict[str, Deque]] = {}
        self.confirmation_counts: Dict[int, Dict[str, int]] = {}
        self.last_normal_posture: Dict[int, float] = {}
        self.active_cheating_events: Dict[int, Dict[str, Dict]] = {}
        self.last_event_time: Dict[int, Dict[str, float]] = {}  # Prevent event spam
        
        # Cleanup old person data periodically for classroom environment
        self.last_cleanup_time = 0
        self.cleanup_interval = 60.0  # Clean up every 60 seconds
        self.last_seen_time: Dict[int, float] = {}  # Track when each person was last seen
        self.max_person_absence = 300.0  # Remove person data after 5 minutes of absence
        self.event_debounce_interval = 2.0  # Minimum 2 seconds between same event types
    
    def update_detection(self, person_id: int, detection_data: Dict[str, Any], timestamp: float) -> List[Dict[str, Any]]:
        """
        Update sliding window with new detection and evaluate rules.
        
        Args:
            person_id: Unique identifier for the person
            detection_data: Detection results with pose and behavior flags
            timestamp: Current timestamp
            
        Returns:
            List of confirmed cheating events
        """
        # Initialize tracking for new person
        if person_id not in self.person_windows:
            self.person_windows[person_id] = {
                'phone_detections': deque(maxlen=self.window_size),
                'head_turn_events': deque(maxlen=self.window_size),
                'head_pitch_events': deque(maxlen=self.window_size),
                'hand_extended_events': deque(maxlen=self.window_size),
                'out_of_frame_events': deque(maxlen=self.window_size),
                'normal_posture_events': deque(maxlen=self.window_size),
                'timestamps': deque(maxlen=self.window_size)
            }
            self.confirmation_counts[person_id] = {}
            self.last_normal_posture[person_id] = timestamp
            self.active_cheating_events[person_id] = {}
        
        # Update sliding window
        windows = self.person_windows[person_id]
        windows['timestamps'].append(timestamp)
        
        # Extract behavior indicators from detection
        phone_detected = detection_data.get('phone_flag', False)
        head_turn_angle = abs(detection_data.get('head_turn_angle', 0.0))
        head_pitch_angle = abs(detection_data.get('lean_angle', 0.0))  # Using lean_angle as pitch proxy
        # Track gesture flag for debugging
        hand_extended = detection_data.get('gesture_flag', False)
        if hand_extended:
            gesture_reason = detection_data.get('gesture_reason', 'unknown')
            self.logger.info(f"🤚 GESTURE DETECTED for person {person_id}: {gesture_reason}")
            self.logger.debug(f"🤚 Full gesture data: {detection_data.get('gesture_details', {})}")
        out_of_frame = detection_data.get('out_of_frame', False)
        
        # Check for normal posture
        is_normal_posture = (
            not phone_detected and 
            head_turn_angle < self.thresholds['normal_head_tilt_threshold'] and
            head_pitch_angle < self.thresholds['normal_head_tilt_threshold'] and
            not hand_extended and
            not out_of_frame
        )
        
        # Update behavior windows
        windows['phone_detections'].append(phone_detected)
        windows['head_turn_events'].append(head_turn_angle > self.thresholds['head_turn_angle_threshold'])
        windows['head_pitch_events'].append(head_pitch_angle > self.thresholds['head_pitch_threshold'])
        windows['hand_extended_events'].append(hand_extended)
        windows['out_of_frame_events'].append(out_of_frame)
        windows['normal_posture_events'].append(is_normal_posture)
        
        # Update normal posture tracking
        if is_normal_posture:
            self.last_normal_posture[person_id] = timestamp
        
        # Check if normal posture has been maintained long enough to reset
        normal_duration = timestamp - self.last_normal_posture[person_id]
        if normal_duration >= self.thresholds['normal_posture_reset_duration']:
            # Reset confirmation counts
            self.confirmation_counts[person_id] = {}
            # Clear active events that should be reset
            events_to_clear = []
            for event_type in self.active_cheating_events[person_id]:
                if event_type in ['head_turn_frequent', 'head_pitch_sustained', 'hand_extended']:
                    events_to_clear.append(event_type)
            for event_type in events_to_clear:
                del self.active_cheating_events[person_id][event_type]
        
        # Update last seen time for this person
        self.last_seen_time[person_id] = timestamp
        
        # Apply research-based rules and generate events
        events = self._evaluate_research_rules(person_id, detection_data, timestamp)
        
        # Perform periodic cleanup for classroom environment
        self._cleanup_absent_persons(timestamp)
        
        return events
    
    def _cleanup_absent_persons(self, current_time: float):
        """Clean up data for persons who haven't been seen for a while in classroom environment."""
        if current_time - self.last_cleanup_time < self.cleanup_interval:
            return
            
        self.last_cleanup_time = current_time
        persons_to_remove = []
        
        # Find persons who haven't been seen for max_person_absence duration
        for person_id in list(self.last_seen_time.keys()):
            if current_time - self.last_seen_time[person_id] > self.max_person_absence:
                persons_to_remove.append(person_id)
        
        # Remove data for absent persons to free memory
        for person_id in persons_to_remove:
            if person_id in self.person_windows:
                del self.person_windows[person_id]
            if person_id in self.confirmation_counts:
                del self.confirmation_counts[person_id]
            if person_id in self.last_normal_posture:
                del self.last_normal_posture[person_id]
            if person_id in self.active_cheating_events:
                del self.active_cheating_events[person_id]
            if person_id in self.last_seen_time:
                del self.last_seen_time[person_id]
        
        if persons_to_remove:
            print(f"🧹 Cleaned up data for {len(persons_to_remove)} absent persons. Active persons: {len(self.person_windows)}")
    
    
    def _evaluate_research_rules(self, person_id: int, detection_data: Dict[str, Any], timestamp: float) -> List[Dict[str, Any]]:
        """Evaluate research-based rules for cheating detection."""
        events = []
        windows = self.person_windows[person_id]
        confirmations = self.confirmation_counts[person_id]
        active_events = self.active_cheating_events[person_id]
        
        # Rule 1: Phone Usage Detection
        phone_events = self._check_phone_usage(windows, confirmations, active_events, person_id, detection_data, timestamp)
        events.extend(phone_events)
        
        # Rule 2: Frequent Head Turning
        head_turn_events = self._check_frequent_head_turning(windows, confirmations, active_events, person_id, detection_data, timestamp)
        events.extend(head_turn_events)
        
        # Rule 3: Sustained Abnormal Head Pitch (Looking Down)
        head_pitch_events = self._check_sustained_head_pitch(windows, confirmations, active_events, person_id, detection_data, timestamp)
        events.extend(head_pitch_events)
        
        # Rule 4: Extended Hand Gestures
        hand_events = self._check_extended_hand_gestures(windows, confirmations, active_events, person_id, detection_data, timestamp)
        events.extend(hand_events)
        
        # Rule 5: Out of Frame Detection
        frame_events = self._check_out_of_frame(windows, confirmations, active_events, person_id, detection_data, timestamp)
        events.extend(frame_events)
        
        return events
    
    def _check_phone_usage(self, windows: Dict, confirmations: Dict, active_events: Dict, 
                          person_id: int, detection_data: Dict, timestamp: float) -> List[Dict[str, Any]]:
        """Check for sustained phone usage (≥3 consecutive frames within window)."""
        events = []
        phone_detections = list(windows['phone_detections'])
        
        if not phone_detections:
            return events
        
        # Count consecutive phone detections at the end of the window
        consecutive_count = 0
        for detection in reversed(phone_detections):
            if detection:
                consecutive_count += 1
            else:
                break
        
        # Check if threshold met
        if consecutive_count >= self.thresholds['phone_consecutive_frames']:
            event_type = 'phone_usage'
            
            # Increment confirmation count
            confirmations[event_type] = confirmations.get(event_type, 0) + 1
            
            # Generate confirmed event if higher phone threshold met
            if confirmations[event_type] >= self.phone_confirmation_threshold:
                if event_type not in active_events:
                    active_events[event_type] = {
                        'first_detected': timestamp,
                        'confirmed': True
                    }
                
                # Calculate sustained duration
                duration = consecutive_count / self.detection_fps
                
                events.append({
                    'timestamp': timestamp,
                    'person_id': f"person_{person_id:03d}",
                    'event_type': 'Phone Usage Detected',
                    'severity': 'red',
                    'confidence': 0.95,
                    'source': 'research_rules',
                    'details': f'Phone detected for {consecutive_count} consecutive frames ({duration:.1f}s)',
                    'bbox': detection_data.get('bbox', [0, 0, 100, 100]),
                    'rule_triggered': 'phone_consecutive_frames',
                    'consecutive_frames': consecutive_count,
                    'duration_seconds': duration
                })
        
        return events
    
    def _check_frequent_head_turning(self, windows: Dict, confirmations: Dict, active_events: Dict,
                                   person_id: int, detection_data: Dict, timestamp: float) -> List[Dict[str, Any]]:
        """Check for frequent head turning (≥3 times in window or sustained >2s)."""
        events = []
        head_turn_events = list(windows['head_turn_events'])
        timestamps = list(windows['timestamps'])
        
        if not head_turn_events or not timestamps:
            return events
        
        # Count head turn events in window
        turn_count = sum(head_turn_events)
        
        # Check for sustained head turning (consecutive frames at end)
        consecutive_turns = 0
        for turn in reversed(head_turn_events):
            if turn:
                consecutive_turns += 1
            else:
                break
        
        sustained_duration = consecutive_turns / self.detection_fps
        
        # Apply research rules
        frequent_turns = turn_count >= self.thresholds['head_turn_frequency_threshold']
        sustained_turns = sustained_duration >= self.thresholds['head_turn_sustained_threshold']
        
        if frequent_turns or sustained_turns:
            event_type = 'head_turn_frequent'
            
            # Increment confirmation count
            confirmations[event_type] = confirmations.get(event_type, 0) + 1
            
            # Generate confirmed event if threshold met and not recently reported
            if confirmations[event_type] >= self.confirmation_threshold:
                # Check debounce timing
                if person_id not in self.last_event_time:
                    self.last_event_time[person_id] = {}
                
                last_reported = self.last_event_time[person_id].get(event_type, 0)
                if timestamp - last_reported < self.event_debounce_interval:
                    return events  # Skip if too recent
                
                if event_type not in active_events:
                    active_events[event_type] = {
                        'first_detected': timestamp,
                        'confirmed': True
                    }
                
                # Update last reported time
                self.last_event_time[person_id][event_type] = timestamp
                
                head_angle = detection_data.get('head_turn_angle', 0.0)
                
                if sustained_turns:
                    severity = 'red'
                    event_name = 'Sustained Head Turning'
                    details = f'Head turning sustained for {sustained_duration:.1f}s (angle: {head_angle:.1f}°)'
                else:
                    severity = 'orange'
                    event_name = 'Frequent Head Turning'
                    details = f'Head turned {turn_count} times in window (current angle: {head_angle:.1f}°)'
                
                events.append({
                    'timestamp': timestamp,
                    'person_id': f"person_{person_id:03d}",
                    'event_type': event_name,
                    'severity': severity,
                    'confidence': 0.8,
                    'source': 'research_rules',
                    'details': details,
                    'bbox': detection_data.get('bbox', [0, 0, 100, 100]),
                    'rule_triggered': 'head_turn_frequency' if frequent_turns else 'head_turn_sustained',
                    'turn_count': turn_count,
                    'sustained_duration': sustained_duration,
                    'head_angle': head_angle
                })
        
        return events
    
    def _check_sustained_head_pitch(self, windows: Dict, confirmations: Dict, active_events: Dict,
                                  person_id: int, detection_data: Dict, timestamp: float) -> List[Dict[str, Any]]:
        """Check for sustained abnormal head pitch (looking down >25° for ≥12 frames)."""
        events = []
        head_pitch_events = list(windows['head_pitch_events'])
        
        if not head_pitch_events:
            return events
        
        # Count consecutive pitch events at end of window
        consecutive_pitch = 0
        for pitch in reversed(head_pitch_events):
            if pitch:
                consecutive_pitch += 1
            else:
                break
        
        # Check if threshold met (≥12 frames ≈ 1.2s)
        if consecutive_pitch >= self.thresholds['head_pitch_frames_threshold']:
            event_type = 'head_pitch_sustained'
            
            # Increment confirmation count
            confirmations[event_type] = confirmations.get(event_type, 0) + 1
            
            # Generate confirmed event if threshold met
            if confirmations[event_type] >= self.confirmation_threshold:
                if event_type not in active_events:
                    active_events[event_type] = {
                        'first_detected': timestamp,
                        'confirmed': True
                    }
                
                duration = consecutive_pitch / self.detection_fps
                pitch_angle = detection_data.get('lean_angle', 0.0)
                
                events.append({
                    'timestamp': timestamp,
                    'person_id': f"person_{person_id:03d}",
                    'event_type': 'Abnormal Looking Down',
                    'severity': 'orange',
                    'confidence': 0.75,
                    'source': 'research_rules',
                    'details': f'Looking down abnormally for {duration:.1f}s (angle: {pitch_angle:.1f}°)',
                    'bbox': detection_data.get('bbox', [0, 0, 100, 100]),
                    'rule_triggered': 'head_pitch_sustained',
                    'consecutive_frames': consecutive_pitch,
                    'duration_seconds': duration,
                    'pitch_angle': pitch_angle
                })
        
        return events
    
    def _check_extended_hand_gestures(self, windows: Dict, confirmations: Dict, active_events: Dict,
                                    person_id: int, detection_data: Dict, timestamp: float) -> List[Dict[str, Any]]:
        """Check for extended hand gestures (≥15 frames ≈ 1.5s)."""
        events = []
        hand_events = list(windows['hand_extended_events'])
        
        if not hand_events:
            return events
        
        # Count consecutive hand events at end of window
        consecutive_hands = 0
        for hand in reversed(hand_events):
            if hand:
                consecutive_hands += 1
            else:
                break
        
        # Check if threshold met (≥10 frames ≈ 1.0s)
        if consecutive_hands >= self.thresholds['hand_extended_frames_threshold']:
            event_type = 'hand_extended'
            
            # Increment confirmation count
            confirmations[event_type] = confirmations.get(event_type, 0) + 1
            
            # Generate confirmed event if threshold met (lower threshold for hand detection)
            if confirmations[event_type] >= self.hand_confirmation_threshold:
                if event_type not in active_events:
                    active_events[event_type] = {
                        'first_detected': timestamp,
                        'confirmed': True
                    }
                
                duration = consecutive_hands / self.detection_fps
                gesture_reason = detection_data.get('gesture_reason', 'unknown_gesture')
                
                events.append({
                    'timestamp': timestamp,
                    'person_id': f"person_{person_id:03d}",
                    'event_type': 'Suspicious Hand Activity',
                    'severity': 'orange',
                    'confidence': 0.7,
                    'source': 'research_rules',
                    'details': f'Hand gesture detected: {gesture_reason} for {duration:.1f}s',
                    'bbox': detection_data.get('bbox', [0, 0, 100, 100]),
                    'rule_triggered': 'hand_extended_duration',
                    'consecutive_frames': consecutive_hands,
                    'duration_seconds': duration,
                    'gesture_type': gesture_reason
                })
        
        return events
    
    def _check_out_of_frame(self, windows: Dict, confirmations: Dict, active_events: Dict,
                          person_id: int, detection_data: Dict, timestamp: float) -> List[Dict[str, Any]]:
        """Check for person going out of frame (≥10 consecutive frames)."""
        events = []
        frame_events = list(windows['out_of_frame_events'])
        
        if not frame_events:
            return events
        
        # Count consecutive out-of-frame events at end of window
        consecutive_out = 0
        for out in reversed(frame_events):
            if out:
                consecutive_out += 1
            else:
                break
        
        # Check if threshold met (≥10 frames)
        if consecutive_out >= self.thresholds['out_of_frame_threshold']:
            event_type = 'out_of_frame'
            
            # Increment confirmation count
            confirmations[event_type] = confirmations.get(event_type, 0) + 1
            
            # Generate confirmed event if threshold met
            if confirmations[event_type] >= self.confirmation_threshold:
                if event_type not in active_events:
                    active_events[event_type] = {
                        'first_detected': timestamp,
                        'confirmed': True
                    }
                
                duration = consecutive_out / self.detection_fps
                
                events.append({
                    'timestamp': timestamp,
                    'person_id': f"person_{person_id:03d}",
                    'event_type': 'Hiding from Camera',
                    'severity': 'red',
                    'confidence': 0.85,
                    'source': 'research_rules',
                    'details': f'Out of frame for {duration:.1f}s',
                    'bbox': detection_data.get('bbox', [0, 0, 100, 100]),
                    'rule_triggered': 'out_of_frame_duration',
                    'consecutive_frames': consecutive_out,
                    'duration_seconds': duration
                })
        
        return events


class EngineHybrid:
    """Research-Based Hybrid Engine with 30 FPS Live Stream + 10 FPS Detection."""
    
    def __init__(self):
        """Initialize the hybrid engine."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing Research-Based Hybrid Engine...")
        
        # Frame rate configuration
        self.skip_rate = 3  # Process every 3rd frame for ~10 FPS detection
        self.target_fps = 30.0
        self.detection_fps = self.target_fps / self.skip_rate  # ~10 FPS
        
        # Device configuration
        self.device = self._setup_device()
        self.logger.info(f"🔧 Using device: {self.device}")
        
        # Detection thresholds
        self.person_conf_thresh = float(os.getenv('PERSON_CONF_THRESH', '0.4'))
        self.phone_conf_thresh = float(os.getenv('PHONE_CONF_THRESH', '0.4'))
        
        # Initialize components
        self._initialize_components()
        
        # Frame tracking
        self.frame_count = 0
        self.last_detections = []
        self.last_tracks = []
        
        # Event state tracking for continuous overlay
        self.active_events = {}  # person_id -> latest event
        self.event_timestamps = {}  # person_id -> timestamp
        self.event_duration = 3.0  # Show events for 3 seconds
        
        # Performance tracking
        self.processing_times = []
        self.detection_times = []
        
        # Video recording
        self.video_recorder = VideoRecorder()
        self.current_session_id: Optional[str] = None
        
        # Evidence storage
        self.evidence_dir = "uploads/evidence"
        self._ensure_evidence_directory()
        
        self.logger.info("✅ Research-Based Hybrid Engine initialization complete")
        self.logger.info(f"📊 Configuration: {self.target_fps} FPS live stream, {self.detection_fps:.1f} FPS detection")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device with GPU-first priority."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
            self.logger.info("🖥️  Using CPU")
        
        return device
    
    def _initialize_components(self):
        """Initialize detection and analysis components."""
        self.logger.info("🔧 Initializing components...")
        
        # YOLOv11 Detector
        self.yolo_detector = YOLO11Detector()
        self.logger.info("✅ YOLOv11 detector ready")
        
        # Pose Detector
        self.pose_detector = PoseDetector()
        self.logger.info("✅ Pose detector ready")
        
        # Simple Tracker with improved settings for single person tracking
        self.tracker = SimpleTracker(max_disappeared=15, max_objects=2)  # Optimized for single-person scenarios
        self.logger.info("✅ Object tracker ready")
        
        # Research-based rule engine
        self.rule_engine = ResearchBasedRuleEngine()
        self.logger.info("✅ Research-based rule engine ready")
        
        # Database manager
        self.db = DBManager()
        self.logger.info("✅ Database manager ready")
    
    def _ensure_evidence_directory(self):
        """Ensure evidence directory exists."""
        try:
            os.makedirs(self.evidence_dir, exist_ok=True)
            self.logger.info(f"📁 Evidence directory: {self.evidence_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create evidence directory: {e}")
            self.evidence_dir = "evidence"
            os.makedirs(self.evidence_dir, exist_ok=True)
    
    def process_frame(self, frame: np.ndarray, cam_id: str = "webcam", 
                     ts: Optional[float] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame with 30 FPS live stream and 10 FPS detection.
        
        Args:
            frame: Input video frame (BGR format)
            cam_id: Camera identifier
            ts: Timestamp (uses current time if None)
            
        Returns:
            Tuple of (overlay_frame, events)
        """
        if ts is None:
            ts = time.time()
        
        start_time = time.time()
        self.frame_count += 1
        all_events = []
        
        try:
            # Determine if this is a detection frame
            is_detection_frame = (self.frame_count % self.skip_rate == 0)
            
            if is_detection_frame:
                # DETECTION FRAME: Run full detection pipeline
                detection_start = time.time()
                
                # Step 1: Object Detection
                detections = self._detect_objects(frame)
                persons = [d for d in detections if d['cls_name'] == 'person']
                phones = [d for d in detections if d['cls_name'] == 'cell phone']
                
                # Filter to keep only the most confident person detection to avoid multiple IDs
                if len(persons) > 1:
                    # Sort by confidence and keep top 2 detections
                    persons = sorted(persons, key=lambda x: x['conf'], reverse=True)[:2]
                    self.logger.debug(f"Filtered to top {len(persons)} person detections")
                
                # Step 2: Update tracker with new detections
                self.last_tracks = self.tracker.update(persons)
                
                # Reset tracker if too many IDs accumulated (single person scenario)
                # More aggressive reset for single-person scenarios
                if len(self.tracker.objects) > 2:  # Reduced from 3 to 2
                    self.logger.warning(f"🔄 Resetting tracker - too many IDs for single person: {len(self.tracker.objects)}")
                    self.tracker = SimpleTracker(max_disappeared=15, max_objects=2)  # Limit to 2 objects max
                    # Re-track with clean tracker
                    self.last_tracks = self.tracker.update(persons)
                
                # Step 3: Pose analysis for tracked persons
                pose_results = self._analyze_poses(frame, self.last_tracks, phones)
                
                # Step 4: Apply research-based rules
                for pose in pose_results:
                    if 'track_id' in pose:
                        person_id = pose['track_id']
                        rule_events = self.rule_engine.update_detection(person_id, pose, ts)
                        all_events.extend(rule_events)
                        
                        # Update active events for continuous overlay
                        for event in rule_events:
                            person_key = event['person_id']
                            self.active_events[person_key] = event
                            self.event_timestamps[person_key] = ts
                
                # Update last detections for overlay (include phones)
                self.last_detections = pose_results + phones
                
                detection_time = time.time() - detection_start
                self.detection_times.append(detection_time)
                
                self.logger.debug(f"🔍 Detection frame {self.frame_count}: {len(persons)} persons detected, "
                               f"{len(self.last_tracks)} tracked, {len(phones)} phones, {len(all_events)} events ({detection_time*1000:.1f}ms)")
                
                # Log if we have too many tracks (debugging)
                if len(self.last_tracks) > 2:
                    track_ids = [t.get('track_id', 'unknown') for t in self.last_tracks]
                    self.logger.warning(f"⚠️ Multiple tracks detected: {track_ids} - may need tracker tuning")
                
            else:
                # NON-DETECTION FRAME: Use tracker interpolation
                if self.last_tracks:
                    # Update tracker without new detections (interpolation)
                    interpolated_tracks = []
                    for track in self.last_tracks:
                        # Simple interpolation: keep same position
                        interpolated_track = track.copy()
                        interpolated_tracks.append(interpolated_track)
                    
                    self.last_tracks = interpolated_tracks
                
                # Use last known detections for overlay
                # No new events generated on non-detection frames
            
            # Clean up expired events for overlay
            self._cleanup_expired_events(ts)
            
            # Get current active events for overlay (includes expired check)
            current_overlay_events = self._get_current_overlay_events()
            
            # Step 5: Create overlay (always at 30 FPS)
            overlay_frame = self._create_overlay(frame, self.last_detections, current_overlay_events)
            
            # Step 6: Record frame (independent 30 FPS recording)
            if self.current_session_id and self.video_recorder.is_recording:
                self.video_recorder.write_frame(overlay_frame)
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Keep recent performance data
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            if len(self.detection_times) > 50:
                self.detection_times.pop(0)
            
            # Log performance metrics periodically
            if self.frame_count % 300 == 0:  # Every 10 seconds at 30 FPS
                avg_fps = 1.0 / np.mean(self.processing_times[-30:]) if self.processing_times else 0
                avg_detection_time = np.mean(self.detection_times[-10:]) if self.detection_times else 0
                self.logger.info(f"📊 Performance: {avg_fps:.1f} FPS, "
                               f"detection: {avg_detection_time*1000:.1f}ms")
            
            return overlay_frame, all_events
            
        except Exception as e:
            self.logger.error(f"Error processing frame {self.frame_count}: {e}")
            return frame, []
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO object detection."""
        try:
            detections = self.yolo_detector.detect(frame)
            
            # Filter by confidence thresholds
            filtered_detections = []
            for det in detections:
                min_conf = self.person_conf_thresh if det['cls_name'] == 'person' else self.phone_conf_thresh
                if det['conf'] >= min_conf:
                    filtered_detections.append(det)
            
            return filtered_detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def _analyze_poses(self, frame: np.ndarray, tracked_persons: List[Dict], 
                      phones: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze poses for tracked persons."""
        try:
            # Get pose estimates
            pose_estimates = self.pose_detector.estimate(frame, phones)
            
            # Match poses with tracked persons
            matched_poses = []
            for track in tracked_persons:
                track_center = self._get_bbox_center(track['bbox'])
                
                # Find closest pose
                best_pose = None
                min_distance = float('inf')
                
                for pose in pose_estimates:
                    pose_center = self._get_bbox_center(pose['bbox'])
                    distance = self._calculate_distance(track_center, pose_center)
                    
                    if distance < min_distance and distance < 100:
                        min_distance = distance
                        best_pose = pose
                
                # Create combined result
                if best_pose:
                    result = best_pose.copy()
                    result['track_id'] = track['track_id']
                    result['person_id'] = f"person_{track['track_id']:03d}"
                    result['bbox'] = track['bbox']  # Use tracker bbox
                    result['detection_conf'] = track['conf']
                else:
                    # Default pose for tracked person without pose detection
                    result = {
                        'track_id': track['track_id'],
                        'person_id': f"person_{track['track_id']:03d}",
                        'bbox': track['bbox'],
                        'detection_conf': track['conf'],
                        'lean_flag': False,
                        'look_flag': False,
                        'phone_flag': False,
                        'gesture_flag': False,
                        'lean_angle': 0.0,
                        'head_turn_angle': 0.0,
                        'confidence': 0.5,
                        'out_of_frame': False
                    }
                
                matched_poses.append(result)
            
            return matched_poses
            
        except Exception as e:
            self.logger.error(f"Pose analysis error: {e}")
            return []
    
    def _cleanup_expired_events(self, current_time: float):
        """Remove expired events from active overlay state."""
        expired_keys = []
        for person_key, timestamp in self.event_timestamps.items():
            if current_time - timestamp > self.event_duration:
                expired_keys.append(person_key)
        
        for key in expired_keys:
            del self.active_events[key]
            del self.event_timestamps[key]
    
    def _get_current_overlay_events(self) -> List[Dict]:
        """Get currently active events for overlay rendering."""
        return list(self.active_events.values())
    
    def _create_overlay(self, frame: np.ndarray, detections: List[Dict], 
                       events: List[Dict]) -> np.ndarray:
        """Create enhanced overlay with confidence-based coloring and false positive mitigation."""
        overlay_frame = frame.copy()
        
        # Simple label mapping for events
        event_labels = {
            'Phone Usage Detected': 'Phone',
            'Frequent Head Turning': 'Turning', 
            'Abnormal Looking Down': 'Looking',
            'Suspicious Hand Activity': 'Hands',
            'Normal': 'Normal'
        }
        
        # Create event lookup for priority-based coloring with confidence tracking
        event_lookup = {}
        event_confidence_map = {}
        for event in events:
            person_id = event['person_id']
            confidence = event.get('confidence', 0.5)
            
            # Priority: red > orange > yellow, but also consider confidence
            if person_id not in event_lookup:
                event_lookup[person_id] = event
                event_confidence_map[person_id] = confidence
            else:
                current_event = event_lookup[person_id]
                current_confidence = event_confidence_map[person_id]
                
                # Replace if higher severity OR same severity with higher confidence
                severity_priority = {'red': 3, 'orange': 2, 'yellow': 1}
                current_priority = severity_priority.get(current_event.get('severity'), 0)
                new_priority = severity_priority.get(event.get('severity'), 0)
                
                if (new_priority > current_priority or 
                    (new_priority == current_priority and confidence > current_confidence)):
                    event_lookup[person_id] = event
                    event_confidence_map[person_id] = confidence
        
        # Separate person and phone detections
        person_detections = [d for d in detections if d.get('cls_name') != 'cell phone']
        phone_detections = [d for d in detections if d.get('cls_name') == 'cell phone']
        
        # Draw person bounding boxes with enhanced confidence-based visualization
        for detection in person_detections:
            if 'bbox' not in detection:
                continue
                
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            person_id = detection.get('person_id', 'unknown')
            detection_conf = detection.get('conf', 0.5)  # Base detection confidence
            keypoint_quality = detection.get('keypoint_quality', 0.5)  # Pose quality from pose detector
            
            # Calculate overall confidence score
            overall_confidence = (detection_conf + keypoint_quality) / 2
            
            # Determine color and visual style based on events and confidence
            color = (0, 255, 0)  # Default green (normal)
            status = "Normal"
            thickness = 1
            alpha = 0.4  # Default lower alpha for normal state
            
            if person_id in event_lookup:
                event = event_lookup[person_id]
                event_confidence = event_confidence_map[person_id]
                severity = event['severity']
                
                # Confidence-based color intensity modulation
                confidence_factor = min(1.0, max(0.3, event_confidence))  # Clamp between 30% and 100%
                
                if severity == 'red':
                    # Red severity - high priority, full intensity at high confidence
                    base_color = (0, 0, 255)
                    color = tuple(int(c * confidence_factor) for c in base_color)
                    thickness = 3 if event_confidence > 0.8 else 2
                    alpha = 0.8 if event_confidence > 0.7 else 0.6
                    
                elif severity == 'orange':
                    # Orange severity - medium priority
                    base_color = (0, 165, 255)
                    color = tuple(int(c * confidence_factor) for c in base_color)
                    thickness = 2 if event_confidence > 0.7 else 1
                    alpha = 0.7 if event_confidence > 0.6 else 0.5
                    
                elif severity == 'yellow':
                    # Yellow severity - low priority, conservative visualization
                    base_color = (0, 255, 255)
                    color = tuple(int(c * confidence_factor) for c in base_color)
                    thickness = 2 if event_confidence > 0.8 else 1
                    alpha = 0.6 if event_confidence > 0.7 else 0.4
                
                # Enhanced status with confidence indicator
                base_status = event_labels.get(event['event_type'], event['event_type'])
                if event_confidence > 0.8:
                    status = f"{base_status}!"  # High confidence indicator
                elif event_confidence > 0.6:
                    status = base_status
                else:
                    status = f"{base_status}?"  # Low confidence indicator
            else:
                # Normal state - modulate by detection quality
                if overall_confidence < 0.4:
                    # Low quality detection - make it more subtle
                    color = (0, 180, 0)  # Darker green
                    alpha = 0.3
                    thickness = 1
                elif overall_confidence > 0.8:
                    # High quality detection - brighter
                    color = (0, 255, 0)
                    alpha = 0.5
                    thickness = 1
            
            # False positive mitigation through visual cues
            # If low overall confidence, add visual indicators
            if overall_confidence < 0.5 and person_id in event_lookup:
                # Draw dashed border for uncertain detections
                self._draw_dashed_rectangle(overlay_frame, (x1, y1), (x2, y2), color, thickness)
            else:
                # Draw solid bounding box
                overlay = overlay_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
            
            # Enhanced label with confidence visualization
            label = status
            font_scale = 0.4
            font_thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Dynamic label background size based on confidence
            bg_height = 18
            bg_width = label_size[0] + 6
            
            # Create confidence-modulated label background
            label_overlay = overlay_frame.copy()
            cv2.rectangle(label_overlay, (x1, y1-bg_height), (x1 + bg_width, y1), (0, 0, 0), -1)
            cv2.rectangle(label_overlay, (x1, y1-bg_height), (x1 + bg_width, y1), color, 1)
            
            # Apply transparency to label background (matched to box alpha)
            cv2.addWeighted(label_overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
            
            # Enhanced text with confidence-based styling
            text_color = (255, 255, 255)  # Default white
            if person_id in event_lookup:
                event_confidence = event_confidence_map[person_id]
                if event_confidence < 0.5:
                    text_color = (200, 200, 200)  # Dimmer for low confidence
            
            cv2.putText(overlay_frame, label, (x1+3, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        
        # Draw phone bounding boxes
        for phone in phone_detections:
            if 'bbox' not in phone:
                continue
                
            x1, y1, x2, y2 = [int(coord) for coord in phone['bbox']]
            
            # Phone detection - always red/critical
            color = (0, 0, 255)  # Red for phone detection
            thickness = 2
            
            # Create overlay for transparency
            overlay = overlay_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Apply transparency to bounding box (0.8 opacity - more visible)
            alpha = 0.8
            cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
            
            # Draw phone label
            label = "Phone"
            font_scale = 0.4
            font_thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Phone label background
            bg_height = 18
            bg_width = label_size[0] + 6
            
            # Create transparent label background
            label_overlay = overlay_frame.copy()
            cv2.rectangle(label_overlay, (x1, y1-bg_height), (x1 + bg_width, y1), (0, 0, 0), -1)
            cv2.rectangle(label_overlay, (x1, y1-bg_height), (x1 + bg_width, y1), color, 1)
            
            # Apply transparency to label background (0.8 opacity)
            label_alpha = 0.8
            cv2.addWeighted(label_overlay, label_alpha, overlay_frame, 1 - label_alpha, 0, overlay_frame)
            
            # White text for phone label
            cv2.putText(overlay_frame, label, (x1+3, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return overlay_frame
    
    def start_session(self, cam_id: str = "webcam", frame_size: Optional[Tuple[int, int]] = None) -> str:
        """Start a new recording session."""
        try:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # Create session in database
            start_timestamp = time.time()
            if not self.db.create_session(session_id, cam_id, start_timestamp):
                self.logger.error("Failed to create session in database")
                return ""
            
            self.current_session_id = session_id
            
            # Start video recording
            if frame_size:
                recording_path = os.path.join("uploads", "recordings", f"{session_id}.mp4")
                if self.video_recorder.start_recording(recording_path, frame_size):
                    self.db.update_session_video_path(session_id, recording_path)
                    self.logger.info(f"🎥 Session {session_id} started with recording: {recording_path}")
                else:
                    self.logger.warning(f"Session {session_id} started but recording failed")
            else:
                self.logger.info(f"📝 Session {session_id} started (no recording)")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            return ""
    
    def stop_session(self) -> dict:
        """Stop the current session."""
        try:
            if not self.current_session_id:
                return {}
            
            session_id = self.current_session_id
            end_timestamp = time.time()
            
            # Stop recording
            if self.video_recorder.is_recording:
                self.video_recorder.stop_recording()
            
            # Update database
            self.db.end_session(session_id, end_timestamp, self.frame_count)
            
            session_info = {
                'session_id': session_id,
                'frame_count': self.frame_count
            }
            
            self.logger.info(f"🏁 Session {session_id} stopped")
            
            # Reset session state
            self.current_session_id = None
            
            return session_info
            
        except Exception as e:
            self.logger.error(f"Failed to stop session: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        avg_fps = 1.0 / np.mean(self.processing_times[-30:]) if self.processing_times else 0
        avg_detection_time = np.mean(self.detection_times[-10:]) if self.detection_times else 0
        
        return {
            'frame_count': self.frame_count,
            'device': str(self.device),
            'performance': {
                'avg_fps': avg_fps,
                'avg_detection_time_ms': avg_detection_time * 1000,
                'target_fps': self.target_fps,
                'detection_fps': self.detection_fps,
                'skip_rate': self.skip_rate
            },
            'rule_engine': {
                'window_size': self.rule_engine.window_size,
                'confirmation_threshold': self.rule_engine.confirmation_threshold,
                'active_persons': len(self.rule_engine.person_windows)
            },
            'thresholds': {
                'person_confidence': self.person_conf_thresh,
                'phone_confidence': self.phone_conf_thresh
            }
        }
    
    def reset(self):
        """Reset engine state."""
        self.logger.info("🔄 Resetting engine state...")
        
        if self.current_session_id:
            self.stop_session()
        
        self.frame_count = 0
        self.processing_times.clear()
        self.detection_times.clear()
        self.last_detections = []
        self.last_tracks = []
        
        # Reset event tracking
        self.active_events = {}
        self.event_timestamps = {}
        
        # Reset tracker
        self.tracker = SimpleTracker(max_disappeared=10, max_objects=2)  # Optimized for single-person scenarios
        
        # Reset rule engine
        self.rule_engine = ResearchBasedRuleEngine()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        self.logger.info("✅ Engine reset complete")
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _draw_dashed_rectangle(self, frame: np.ndarray, pt1: Tuple[int, int], 
                              pt2: Tuple[int, int], color: Tuple[int, int, int], 
                              thickness: int = 1, dash_length: int = 10):
        """Draw a dashed rectangle to indicate uncertain detections."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw dashed lines for each side of rectangle
        def draw_dashed_line(start, end, is_horizontal=True):
            if is_horizontal:
                length = abs(end[0] - start[0])
                step_x = dash_length if end[0] > start[0] else -dash_length
                for i in range(0, length, dash_length * 2):
                    line_start = (start[0] + i, start[1])
                    line_end = (min(start[0] + i + dash_length, end[0]) if step_x > 0 
                               else max(start[0] + i - dash_length, end[0]), start[1])
                    cv2.line(frame, line_start, line_end, color, thickness)
            else:
                length = abs(end[1] - start[1])
                step_y = dash_length if end[1] > start[1] else -dash_length
                for i in range(0, length, dash_length * 2):
                    line_start = (start[0], start[1] + i)
                    line_end = (start[0], min(start[1] + i + dash_length, end[1]) if step_y > 0 
                               else max(start[1] + i - dash_length, end[1]))
                    cv2.line(frame, line_start, line_end, color, thickness)
        
        # Draw four dashed sides
        draw_dashed_line((x1, y1), (x2, y1), True)   # Top
        draw_dashed_line((x2, y1), (x2, y2), False)  # Right
        draw_dashed_line((x2, y2), (x1, y2), True)   # Bottom
        draw_dashed_line((x1, y2), (x1, y1), False)  # Left
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
import supervision as sv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from ..detectors.yolo11_detector import YOLO11Detector
from ..detectors.pose_detector import PoseDetector
from ..db.db_manager import DBManager
from ..video_recorder import VideoRecorder

# ============================================================================
# DEPRECATED: SimpleTracker - Now using Supervision's ByteTrack
# Kept for reference only. ByteTrack provides superior tracking with:
# - Better ID persistence across occlusions
# - Robust handling of crowded scenes
# - Optimized matching algorithms
# ============================================================================
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
        self.assignment_threshold = 150  # Reduced from 250 to 150 for better multi-person tracking in classroom
    
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
        
        # Research-based thresholds for classroom behaviors (3 detections only)
        self.thresholds = {
            # Phone Usage: Detected phone in ≥10 consecutive frames (realistic for genuine use)
            # At 10 FPS detection, 10 frames = 1.0 second of continuous detection
            'phone_consecutive_frames': 10,  # Require 1 second of sustained phone use - realistic for classroom
            'phone_duration_threshold': 1.5,  # seconds - longer duration requirement
            
            # Looking Away Frequently: Head yaw >40° left/right, occurring ≥2 times in 4s or held >2s  
            'head_turn_angle_threshold': 40.0,  # degrees - increased threshold for more deliberate turns
            'head_turn_frequency_threshold': 2,  # occurrences in window - sensitive to repeated looking 
            'head_turn_sustained_threshold': 2.0,  # seconds - quick detection of sustained looking
            
            # Hand Extended: ≥10 frames (~1.0s) - higher temporal smoothing to reduce sensitivity
            'hand_extended_frames_threshold': 10,  # frames - less sensitive for hand detection
            
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
        self.previous_head_turn_state: Dict[int, bool] = {}  # REMOVED - now using instant detection like hand extensions
        
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
                'hand_extended_events': deque(maxlen=self.window_size),
                'normal_posture_events': deque(maxlen=self.window_size),
                'timestamps': deque(maxlen=self.window_size)
            }
            self.confirmation_counts[person_id] = {}
            self.last_normal_posture[person_id] = timestamp
            self.active_cheating_events[person_id] = {}
            # REMOVED head turn state tracking - now using instant detection like hand extensions
        
        # Update sliding window
        windows = self.person_windows[person_id]
        windows['timestamps'].append(timestamp)
        
        # Extract behavior indicators from detection
        phone_detected = detection_data.get('phone_flag', False)
        head_turn_angle = abs(detection_data.get('head_turn_angle', 0.0))
        # Track gesture flag for debugging - FOCUS ON SIDEWARD EXTENSIONS ONLY
        hand_extended = detection_data.get('gesture_flag', False)
        gesture_reason = detection_data.get('gesture_reason', 'unknown')
        
        # Only flag sideward hand extensions (not face covering)
        is_sideward_gesture = hand_extended and (
            'sideward' in gesture_reason.lower() or 
            'reach' in gesture_reason.lower()
        )
        
        if hand_extended:
            if is_sideward_gesture:
                # Log gesture detection at DEBUG level to reduce noise
                self.logger.debug(f"🤚 SIDEWARD GESTURE DETECTED for person {person_id}: {gesture_reason}")
                self.logger.debug(f"🤚 Full sideward gesture data: {detection_data.get('gesture_details', {})}")
            else:
                self.logger.debug(f"🤚 Ignoring non-sideward gesture for person {person_id}: {gesture_reason} (likely face covering)")
                hand_extended = False  # Override - don't count face covering as suspicious
        
        # Check for normal posture (only 3 detections matter now)
        is_normal_posture = (
            not phone_detected and 
            head_turn_angle < self.thresholds['normal_head_tilt_threshold'] and
            not hand_extended
        )
        
        # Update behavior windows
        windows['phone_detections'].append(phone_detected)
        
        # Head turn event detection - INSTANT DETECTION like hand extensions
        current_head_turn = head_turn_angle >= self.thresholds['head_turn_angle_threshold']
        
        # Log head turn detection for debugging
        if current_head_turn:
            direction = "RIGHT" if detection_data.get('head_turn_angle', 0.0) > 0 else "LEFT"
            self.logger.info(f"🔄 HEAD TURN DETECTED for person {person_id}: {direction} turn ({head_turn_angle:.1f}°)")
        
        # Add instant head turn detection to windows (like hand extensions)
        windows['head_turn_events'].append(current_head_turn)
        
        windows['hand_extended_events'].append(hand_extended)
        windows['normal_posture_events'].append(is_normal_posture)
        
        # Update normal posture tracking
        if is_normal_posture:
            self.last_normal_posture[person_id] = timestamp
        
        # Check if normal posture has been maintained long enough to reset
        normal_duration = timestamp - self.last_normal_posture[person_id]
        if normal_duration >= self.thresholds['normal_posture_reset_duration']:
            # Reset confirmation counts
            self.confirmation_counts[person_id] = {}
            # Clear active events that should be reset (only 3 detections)
            events_to_clear = []
            for event_type in self.active_cheating_events[person_id]:
                if event_type in ['head_turn_instant', 'hand_extended', 'phone_usage']:
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
        """Evaluate research-based rules for cheating detection - 3 DETECTIONS ONLY."""
        events = []
        windows = self.person_windows[person_id]
        confirmations = self.confirmation_counts[person_id]
        active_events = self.active_cheating_events[person_id]
        
        # ONLY 3 DETECTION RULES (as requested by user):
        
        # Rule 1: Phone Usage Detection
        phone_events = self._check_phone_usage(windows, confirmations, active_events, person_id, detection_data, timestamp)
        events.extend(phone_events)
        
        # Rule 2: Head Turning Left and Right
        head_turn_events = self._check_frequent_head_turning(windows, confirmations, active_events, person_id, detection_data, timestamp)
        events.extend(head_turn_events)
        
        # Rule 3: Hand Extension (sideward reaching gestures)
        hand_events = self._check_extended_hand_gestures(windows, confirmations, active_events, person_id, detection_data, timestamp)
        events.extend(hand_events)
        
        # DISABLED RULES (as requested):
        # - Sustained Abnormal Head Pitch (Looking Down) - REMOVED
        # - Out of Frame Detection - REMOVED
        
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
                    'event_type': 'Phone Usage',  # Simple label for bounding box
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
        """Check for INSTANT head turning detection - no waiting for multiple instances."""
        events = []
        head_turn_events = list(windows['head_turn_events'])
        timestamps = list(windows['timestamps'])
        
        if not head_turn_events or not timestamps:
            return events
        
        # INSTANT DETECTION: Check if current frame has head turning based on angle
        head_angle = abs(detection_data.get('head_turn_angle', 0.0))
        raw_head_angle = detection_data.get('head_turn_angle', 0.0)  # Keep sign for direction
        
        # INSTANT HEAD TURN DETECTION - enhanced sensitivity for classroom monitoring
        if head_angle >= 30.0:  # 30+ degree turn for enhanced sensitivity (inclusive)
            event_type = 'head_turn_instant'
            
            # Check debounce timing to avoid spam
            if person_id not in self.last_event_time:
                self.last_event_time[person_id] = {}
            
            last_reported = self.last_event_time[person_id].get(event_type, 0)
            if timestamp - last_reported < 0.5:  # Enhanced responsiveness - 0.5 second debounce for sensitive detection
                return events  # Skip if too recent
            
            if event_type not in active_events:
                active_events[event_type] = {
                    'first_detected': timestamp,
                    'confirmed': True
                }
            
            # Update last reported time
            self.last_event_time[person_id][event_type] = timestamp
            
            # Determine turn direction correctly
            direction = "RIGHT" if raw_head_angle > 0 else "LEFT"
            severity = 'orange'
            event_name = 'Head Turning'  # Clear label for detection
            details = f'Head turned {direction} ({head_angle:.1f}° detected instantly)'
            
            events.append({
                'timestamp': timestamp,
                'person_id': f"person_{person_id:03d}",
                'event_type': event_name,
                'severity': severity,
                'confidence': 0.8,
                'source': 'research_rules',
                'details': details,
                'bbox': detection_data.get('bbox', [0, 0, 100, 100]),
                'rule_triggered': 'instant_head_turn',
                'turn_direction': direction,
                'head_angle': head_angle
            })
        
        return events
    
    # UNUSED METHOD - Detection disabled (only Phone, Head Turn, Hand Extension active)
    # def _check_sustained_head_pitch(self, windows: Dict, confirmations: Dict, active_events: Dict,
    #                               person_id: int, detection_data: Dict, timestamp: float) -> List[Dict[str, Any]]:
    #     """Check for sustained abnormal head pitch (looking down >25° for ≥12 frames)."""
    #     events = []
    #     head_pitch_events = list(windows['head_pitch_events'])
    #     
    #     if not head_pitch_events:
    #         return events
    #     
    #     # Count consecutive pitch events at end of window
    #     consecutive_pitch = 0
    #     for pitch in reversed(head_pitch_events):
    #         if pitch:
    #             consecutive_pitch += 1
    #         else:
    #             break
    #     
    #     # Check if threshold met (≥12 frames ≈ 1.2s)
    #     if consecutive_pitch >= self.thresholds['head_pitch_frames_threshold']:
    #         event_type = 'head_pitch_sustained'
    #         
    #         # Increment confirmation count
    #         confirmations[event_type] = confirmations.get(event_type, 0) + 1
    #         
    #         # Generate confirmed event if threshold met
    #         if confirmations[event_type] >= self.confirmation_threshold:
    #             if event_type not in active_events:
    #                 active_events[event_type] = {
    #                     'first_detected': timestamp,
    #                     'confirmed': True
    #                 }
    #             
    #             duration = consecutive_pitch / self.detection_fps
    #             pitch_angle = detection_data.get('lean_angle', 0.0)
    #             
    #             events.append({
    #                 'timestamp': timestamp,
    #                 'person_id': f"person_{person_id:03d}",
    #                 'event_type': 'Abnormal Looking Down',
    #                 'severity': 'orange',
    #                 'confidence': 0.75,
    #                 'source': 'research_rules',
    #                 'details': f'Looking down abnormally for {duration:.1f}s (angle: {pitch_angle:.1f}°)',
    #                 'bbox': detection_data.get('bbox', [0, 0, 100, 100]),
    #                 'rule_triggered': 'head_pitch_sustained',
    #                 'consecutive_frames': consecutive_pitch,
    #                 'duration_seconds': duration,
    #                 'pitch_angle': pitch_angle
    #             })
    #     
    #     return events
    
    def _check_extended_hand_gestures(self, windows: Dict, confirmations: Dict, active_events: Dict,
                                    person_id: int, detection_data: Dict, timestamp: float) -> List[Dict[str, Any]]:
        """Check for sustained sideward hand gestures - CLASSROOM FOCUSED."""
        events = []
        
        # Check if current frame has sideward gesture
        gesture_flag = detection_data.get('gesture_flag', False)
        gesture_reason = detection_data.get('gesture_reason', 'unknown')
        
        if not gesture_flag:
            # Clear any tracking if no gesture detected
            if hasattr(self, 'gesture_tracking'):
                self.gesture_tracking.pop(person_id, None)
            return events
        
        # Only process sideward extensions (ignore face covering)
        is_sideward = 'sideward' in gesture_reason.lower() or 'reach' in gesture_reason.lower()
        
        if not is_sideward:
            self.logger.debug(f"🤚 Ignoring non-sideward hand gesture: {gesture_reason}")
            return events
        
        # Initialize gesture tracking
        if not hasattr(self, 'gesture_tracking'):
            self.gesture_tracking = {}
        
        if person_id not in self.gesture_tracking:
            self.gesture_tracking[person_id] = {}
        
        # Track gesture persistence
        gesture_key = f"{gesture_reason}"
        current_tracking = self.gesture_tracking[person_id].get(gesture_key, {
            'first_detected': timestamp,
            'last_seen': timestamp,
            'frames_count': 0,
            'confirmed': False
        })
        
        # Update tracking
        current_tracking['last_seen'] = timestamp
        current_tracking['frames_count'] += 1
        
        # Require gesture to be sustained for at least 0.5 seconds (about 8-10 frames at 15-20 FPS)
        sustained_duration = timestamp - current_tracking['first_detected']
        min_duration = 0.5  # seconds
        min_frames = 8  # minimum frames
        
        self.gesture_tracking[person_id][gesture_key] = current_tracking
        
        # Only trigger if gesture is sustained AND meets minimum requirements
        if sustained_duration >= min_duration and current_tracking['frames_count'] >= min_frames:
            if not current_tracking['confirmed']:
                # Mark as confirmed and check debounce
                event_type = 'hand_extended'
                
                # Add debounce timing to prevent spam
                if person_id not in self.last_event_time:
                    self.last_event_time[person_id] = {}
                
                last_reported = self.last_event_time[person_id].get(event_type, 0)
                if timestamp - last_reported < 3.0:  # 3-second debounce
                    return events  # Skip if too recent
                
                # Mark as confirmed and generate event
                current_tracking['confirmed'] = True
                self.last_event_time[person_id][event_type] = timestamp
                
                # Generate sustained gesture event
                active_events[event_type] = {
                    'first_detected': current_tracking['first_detected'],
                    'confirmed': True
                }
                
                events.append({
                    'timestamp': timestamp,
                    'person_id': f"person_{person_id:03d}",
                    'event_type': 'Hand Extension',  # Simple label for bounding box
                    'severity': 'orange',
                    'confidence': 0.85,  # Higher confidence for sustained gestures
                    'source': 'research_rules',
                    'details': f'Sustained sideward hand extension: {gesture_reason} (sustained for {sustained_duration:.1f}s)',
                    'bbox': detection_data.get('bbox', [0, 0, 100, 100]),
                    'rule_triggered': 'hand_extended_sustained',
                    'gesture_type': gesture_reason,
                    'duration': sustained_duration,
                    'frame_count': current_tracking['frames_count']
                })
                
                self.logger.info(f"🤚 SUSTAINED SIDEWARD GESTURE DETECTED for person {person_id}: {gesture_reason} (sustained {sustained_duration:.1f}s, {current_tracking['frames_count']} frames)")
        
        return events
    
    # UNUSED METHOD - Detection disabled (only Phone, Head Turn, Hand Extension active)
    # def _check_out_of_frame(self, windows: Dict, confirmations: Dict, active_events: Dict,
    #                       person_id: int, detection_data: Dict, timestamp: float) -> List[Dict[str, Any]]:
    #     """Check for person going out of frame (≥10 consecutive frames)."""
    #     events = []
    #     frame_events = list(windows['out_of_frame_events'])
    #     
    #     if not frame_events:
    #         return events
    #     
    #     # Count consecutive out-of-frame events at end of window
    #     consecutive_out = 0
    #     for out in reversed(frame_events):
    #         if out:
    #             consecutive_out += 1
    #         else:
    #             break
    #     
    #     # Check if threshold met (≥10 frames)
    #     if consecutive_out >= self.thresholds['out_of_frame_threshold']:
    #         event_type = 'out_of_frame'
    #         
    #         # Increment confirmation count
    #         confirmations[event_type] = confirmations.get(event_type, 0) + 1
    #         
    #         # Generate confirmed event if threshold met
    #         if confirmations[event_type] >= self.confirmation_threshold:
    #             if event_type not in active_events:
    #                 active_events[event_type] = {
    #                     'first_detected': timestamp,
    #                     'confirmed': True
    #                 }
    #             
    #             duration = consecutive_out / self.detection_fps
    #             
    #             events.append({
    #                 'timestamp': timestamp,
    #                 'person_id': f"person_{person_id:03d}",
    #                 'event_type': 'Hiding from Camera',
    #                 'severity': 'red',
    #                 'confidence': 0.85,
    #                 'source': 'research_rules',
    #                 'details': f'Out of frame for {duration:.1f}s',
    #                 'bbox': detection_data.get('bbox', [0, 0, 100, 100]),
    #                 'rule_triggered': 'out_of_frame_duration',
    #                 'consecutive_frames': consecutive_out,
    #                 'duration_seconds': duration
    #             })
    #     
    #     return events


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
        
        # Detection thresholds - optimized for classroom multi-student detection
        self.person_conf_thresh = float(os.getenv('PERSON_CONF_THRESH', '0.25'))  # CLASSROOM MODE: Lower threshold for better detection (was 0.40)
        self.phone_conf_thresh = float(os.getenv('PHONE_CONF_THRESH', '0.30'))  # REALISTIC: Balanced threshold to avoid false positives
        
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
        
        # Supervision ByteTrack for robust multi-person tracking
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.35,  # Higher threshold to reduce false positives (was 0.25)
            lost_track_buffer=60,  # Longer buffer to maintain IDs (2 seconds at 30 FPS)
            minimum_matching_threshold=0.6,  # Lower IOU threshold for better matching (was 0.8)
            frame_rate=30  # Match your target FPS
        )
        self.logger.info("✅ Supervision ByteTrack ready (optimized for ID persistence & NMS)")
        
        # Research-based rule engine
        self.rule_engine = ResearchBasedRuleEngine()
        self.logger.info("✅ Research-based rule engine ready")
        
        # Database manager
        self.db = DBManager()
        self.logger.info("✅ Database manager ready")
        
        # Alarm system
        self._initialize_alarm_system()
        self.logger.info("🔊 Alarm system initialized with sound file")
    
    def _ensure_evidence_directory(self):
        """Ensure evidence directory exists."""
        try:
            os.makedirs(self.evidence_dir, exist_ok=True)
            self.logger.info(f"📁 Evidence directory: {self.evidence_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create evidence directory: {e}")
            self.evidence_dir = "evidence"
            os.makedirs(self.evidence_dir, exist_ok=True)
    
    def _initialize_alarm_system(self):
        """Initialize pygame mixer for alarm sounds."""
        try:
            import pygame
            pygame.mixer.init()
            
            # Try to load alarm sound
            sound_files = ["alarm.wav", "RIZZ Sound Effect.wav", "alarm.mp3"]
            self.alarm_sound = None
            
            for sound_file in sound_files:
                try:
                    self.alarm_sound = pygame.mixer.Sound(sound_file)
                    break
                except:
                    continue
                    
            if self.alarm_sound is None:
                self.logger.warning("⚠️ No alarm sound file found, using system beep")
                
        except ImportError:
            self.logger.warning("⚠️ pygame not available, alarm sounds disabled")
            self.alarm_sound = None
    
    def _trigger_alarm(self, event_type: str):
        """Trigger alarm sound for phone detection only."""
        if event_type == "Phone Usage":  # Fixed to match actual event type
            try:
                if self.alarm_sound:
                    self.alarm_sound.play()
                    self.logger.info(f"🚨 ALARM TRIGGERED: {event_type}")
                else:
                    # Fallback to system beep
                    import winsound
                    winsound.Beep(1000, 500)  # 1000 Hz for 500ms
                    self.logger.info(f"🚨 BEEP ALARM TRIGGERED: {event_type}")
            except Exception as e:
                self.logger.debug(f"Alarm failed: {e}")
        # No alarm for other event types (head turning, looking down, gestures)
    
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
                
                # Filter to keep high-confidence person detections for classroom
                persons = [p for p in persons if p['conf'] >= self.person_conf_thresh]
                # Sort by confidence but keep all detections (no artificial limit)
                persons = sorted(persons, key=lambda x: x['conf'], reverse=True)
                
                # Step 1.5: Apply NMS (Non-Maximum Suppression) to remove overlapping detections
                if len(persons) > 1:
                    # Convert to numpy arrays for NMS
                    boxes = np.array([p['bbox'] for p in persons])
                    scores = np.array([p['conf'] for p in persons])
                    
                    # Simple NMS implementation
                    keep_indices = self._nms(boxes, scores, iou_threshold=0.5)
                    
                    # Keep only non-overlapping detections
                    persons_before = len(persons)
                    persons = [persons[i] for i in keep_indices]
                    
                    if len(persons) < persons_before:
                        self.logger.debug(f"🔧 NMS: Removed {persons_before - len(persons)} overlapping detections")
                
                # Step 2: Update tracker with new detections using Supervision ByteTrack
                if len(persons) > 0:
                    detections_sv = sv.Detections(
                        xyxy=np.array([p['bbox'] for p in persons]),
                        confidence=np.array([p['conf'] for p in persons]),
                        class_id=np.array([0] * len(persons))  # person class
                    )
                    
                    # Update ByteTrack tracker
                    detections_sv = self.tracker.update_with_detections(detections_sv)
                    
                    # Convert back to our format with persistent track IDs from ByteTrack
                    self.last_tracks = []
                    for i, (bbox, conf, track_id) in enumerate(zip(
                        detections_sv.xyxy, 
                        detections_sv.confidence, 
                        detections_sv.tracker_id
                    )):
                        track_obj = persons[i].copy()
                        track_obj['track_id'] = int(track_id)
                        track_obj['bbox'] = bbox.tolist()
                        self.last_tracks.append(track_obj)
                    
                    # Simple log showing detected person IDs (only on changes)
                    track_ids = [t['track_id'] for t in self.last_tracks]
                    
                    # Only log if person count or IDs changed
                    if not hasattr(self, '_last_logged_tracks') or self._last_logged_tracks != track_ids:
                        self.logger.info(f"👥 Detected {len(self.last_tracks)} persons: {track_ids}")
                        self._last_logged_tracks = track_ids
                else:
                    # No persons detected - update with empty detections
                    empty_detections = sv.Detections.empty()
                    self.tracker.update_with_detections(empty_detections)
                    self.last_tracks = []
                
                # Step 3: Pose analysis for tracked persons
                pose_results = self._analyze_poses(frame, self.last_tracks, phones)
                
                # Step 4: Apply research-based rules
                for pose in pose_results:
                    if 'track_id' in pose:
                        person_id = pose['track_id']
                        rule_events = self.rule_engine.update_detection(person_id, pose, ts)
                        all_events.extend(rule_events)
                        
                        # Log events when generated
                        if rule_events:
                            for event in rule_events:
                                self.logger.info(f"🚨 {event['event_type']}: {event['person_id']}")
                        
                        # Trigger alarms for phone detection only
                        for event in rule_events:
                            self._trigger_alarm(event.get('event_type', ''))
                        
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
                
                # Log if we have excessive tracks (debugging for classroom)
                if len(self.last_tracks) > 15:
                    track_ids = [t.get('track_id', 'unknown') for t in self.last_tracks]
                    self.logger.warning(f"⚠️ Many tracks detected: {len(track_ids)} students - performance may be affected")
                
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
                
                # Create combined result WITH ByteTrack track_id
                if best_pose:
                    result = best_pose.copy()
                    result['track_id'] = track['track_id']  # ← ByteTrack ID preserved here
                    result['person_id'] = f"person_{track['track_id']:03d}"
                    result['bbox'] = track['bbox']  # Use tracker bbox
                    result['detection_conf'] = track['conf']
                    
                    # Log phone detection with proper track ID
                    if result.get('phone_flag', False):
                        self.logger.info(f"📱 PHONE DETECTED: Track ID {track['track_id']} (Person {result['person_id']})")
                else:
                    # Default pose for tracked person without pose detection
                    result = {
                        'track_id': track['track_id'],  # ← ByteTrack ID preserved here
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
        
        # Simple label mapping for events (3 detections only)
        event_labels = {
            'Phone Usage': 'Phone',
            'Phone Usage Detected': 'Phone',
            'Head Turning': 'Turning',
            'Frequent Head Turning': 'Turning',
            'Hand Extension': 'Hands',
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
        
        # FIRST: Draw unmatched phone detections (phones NOT held by anyone)
        # These appear as standalone red boxes for phones on desk/table
        person_ids_with_phones = set()
        for event in events:
            if event.get('event_type') == 'Phone Usage':
                person_ids_with_phones.add(event.get('person_id'))
        
        # Draw phone bounding boxes ONLY for unmatched phones (not held by tracked persons)
        for phone in phone_detections:
            if 'bbox' not in phone:
                continue
            
            # Check if this phone is already matched to a person with phone event
            phone_matched = False
            phone_bbox = phone['bbox']
            
            for detection in person_detections:
                if 'bbox' not in detection:
                    continue
                person_id = detection.get('person_id', 'unknown')
                
                # Skip if this person has phone event (phone already highlighted via person box)
                if person_id in person_ids_with_phones:
                    person_bbox = detection['bbox']
                    # Check if phone is near this person
                    px1, py1, px2, py2 = person_bbox
                    phx1, phy1, phx2, phy2 = phone_bbox
                    
                    # Simple overlap check
                    if not (phx2 < px1 or phx1 > px2 or phy2 < py1 or phy1 > py2):
                        phone_matched = True
                        break
            
            # Only draw standalone phone boxes for unmatched phones
            if not phone_matched:
                x1, y1, x2, y2 = [int(coord) for coord in phone_bbox]
                
                # Unmatched phone - draw with medium visibility
                color = (0, 0, 255)  # Red for unmatched phones (potential cheating tool)
                thickness = 2
                
                # Create overlay for transparency
                overlay = overlay_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                
                # Apply medium transparency
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
                
                # Label for unmatched phone
                label = "Phone"
                font_scale = 0.4
                font_thickness = 1
                
                # Add label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                label_bg_height = 18
                label_bg_width = label_size[0] + 6
                
                # Draw label background
                cv2.rectangle(overlay_frame, (x1, y1-label_bg_height), (x1 + label_bg_width, y1), (0, 0, 0), -1)
                cv2.rectangle(overlay_frame, (x1, y1-label_bg_height), (x1 + label_bg_width, y1), color, 1)
                
                # Draw text
                cv2.putText(overlay_frame, label, (x1+3, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # SECOND: Draw person bounding boxes with enhanced confidence-based visualization
        # Person boxes with phone events will show as red boxes (phone usage)
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
                    # Red severity - high priority, reduced opacity for less intrusiveness
                    base_color = (0, 0, 255)
                    color = tuple(int(c * confidence_factor) for c in base_color)
                    thickness = 3 if event_confidence > 0.8 else 2
                    alpha = 0.35 if event_confidence > 0.7 else 0.25  # Reduced opacity from 0.8/0.6 to 0.35/0.25
                    
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
            font_scale = 0.5  # Increased from 0.4 for better readability
            font_thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Dynamic label background size based on confidence
            bg_height = 20  # Increased from 18 for better padding
            bg_width = label_size[0] + 8  # Increased padding from 6 to 8
            
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
            
            # Position text in center of label background
            text_y = y1 - 6  # Adjusted for larger font
            cv2.putText(overlay_frame, label, (x1+4, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        
        # Phone boxes already drawn at the top (only unmatched phones)
        # Person boxes with phone events show as RED boxes with "Phone" label
        
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
        
        # Reset ByteTrack tracker for multi-student scenarios
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        
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
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
        """
        Apply Non-Maximum Suppression to remove overlapping bounding boxes.
        
        Args:
            boxes: Array of bounding boxes [x1, y1, x2, y2]
            scores: Array of confidence scores
            iou_threshold: IoU threshold for suppression (0.5 = 50% overlap)
        
        Returns:
            List of indices to keep
        """
        if len(boxes) == 0:
            return []
        
        # Convert to float
        boxes = boxes.astype(np.float32)
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence score (descending)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            # Keep the box with highest score
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            # Calculate intersection area
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            # Calculate IoU
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU less than threshold
            indices = np.where(iou <= iou_threshold)[0]
            order = order[indices + 1]
        
        return keep
    
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
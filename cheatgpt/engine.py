"""Core engine for coordinating the complete CheatGPT3 detection pipeline.

This module orchestrates all components:
- YOLOv11 detection (persons, phones)
- Object tracking for ID consistency
- Pose detection and behavior analysis
- Policy rule evaluation
- Evidence saving and visualization
"""

import os
import time
import logging
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
from datetime import datetime

from .detectors.yolo11_detector import YOLO11Detector
from .detectors.pose_detector import PoseDetector
from .detectors.tracker import Tracker
from .policy.rules import check_rule, get_active_violations, Severity
from .db.db_manager import DBManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Engine:
    """Main processing engine that coordinates all detection and analysis components."""
    
    def __init__(self):
        """Initialize the complete detection pipeline."""
        logger.info("Initializing CheatGPT3 Engine...")
        
        # Debug configuration
        self.debug_mode = os.getenv('DEBUG_ENGINE', 'false').lower() == 'true'
        
        # Initialize components
        self.yolo_detector = YOLO11Detector()
        self.pose_detector = PoseDetector()
        self.tracker = Tracker(max_disappeared=30, max_distance=100.0)
        self.db = DBManager()
        
        # Evidence storage
        self.evidence_dir = "uploads/evidence"
        self._ensure_evidence_directory()
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        
        logger.info("CheatGPT3 Engine initialization complete")
        logger.info(f"Evidence directory: {os.path.abspath(self.evidence_dir)}")
        logger.info(f"Debug mode: {self.debug_mode}")
    
    def _ensure_evidence_directory(self):
        """Ensure evidence directory exists."""
        try:
            os.makedirs(self.evidence_dir, exist_ok=True)
            logger.info(f"Evidence directory ready: {self.evidence_dir}")
        except Exception as e:
            logger.error(f"Failed to create evidence directory: {e}")
            # Fallback to current directory
            self.evidence_dir = "evidence"
            os.makedirs(self.evidence_dir, exist_ok=True)
    
    def process_frame(self, frame: np.ndarray, cam_id: str = "cam_01", 
                     ts: Optional[float] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame through the complete detection pipeline.
        
        Args:
            frame: Input video frame (BGR format)
            cam_id: Camera identifier
            ts: Timestamp (uses current time if None)
            
        Returns:
            Tuple of (overlay_frame, events) where:
            - overlay_frame: Frame with bounding boxes and labels
            - events: List of violation events detected
        """
        if ts is None:
            ts = time.time()
        
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # Step 1: YOLOv11 Detection (persons, phones)
            logger.debug(f"Frame {self.frame_count}: Running YOLO detection...")
            detections = self.yolo_detector.detect(frame)
            
            persons = [det for det in detections if det['cls_name'] == 'person']
            phones = [det for det in detections if det['cls_name'] == 'cell phone']
            
            logger.debug(f"YOLO detected {len(persons)} persons, {len(phones)} phones")
            
            # Step 2: Tracker Updates IDs
            logger.debug(f"Frame {self.frame_count}: Updating tracker...")
            tracked_persons = self.tracker.update(persons, ts)
            
            logger.debug(f"Tracker assigned IDs to {len(tracked_persons)} persons")
            
            # Step 3: Pose Detector Extracts Behavior Features
            logger.debug(f"Frame {self.frame_count}: Analyzing poses...")
            pose_estimates = self.pose_detector.estimate(frame, phones)
            
            # Merge tracking IDs with pose estimates
            merged_poses = self._merge_tracking_and_poses(tracked_persons, pose_estimates)
            
            logger.debug(f"Pose analysis complete for {len(merged_poses)} persons")
            
            # Step 4: Policy Module Applies Rules
            logger.debug(f"Frame {self.frame_count}: Applying policy rules...")
            events = []
            
            for pose in merged_poses:
                if self.debug_mode:
                    logger.info(f"Evaluating behavior for {pose['track_id']}: "
                               f"lean={pose['lean_flag']}, look={pose['look_flag']}, phone={pose['phone_flag']}")
                
                violations = check_rule(
                    pose['track_id'],
                    pose['lean_flag'],
                    pose['look_flag'],
                    pose['phone_flag'],
                    ts
                )
                
                # Convert violations to events
                for violation in violations:
                    event = {
                        'timestamp': ts,
                        'cam_id': cam_id,
                        'track_id': violation.track_id,
                        'event_type': violation.label,
                        'severity': violation.severity.label,
                        'confidence': violation.confidence,
                        'bbox': pose['bbox'],
                        'details': violation.details
                    }
                    events.append(event)
                    
                    if self.debug_mode:
                        logger.info(f"🚨 VIOLATION: {event['event_type']} for {event['track_id']} - {event['details']}")
            
            logger.debug(f"Policy evaluation found {len(events)} events")
            
            # Step 5: Save Evidence for Cheating Events
            cheating_events = [e for e in events if e['severity'] == 'Cheating']
            if cheating_events:
                self._save_evidence_frame(frame, cheating_events, cam_id, ts)
            
            # Step 6: Draw Bounding Boxes with Labels and Severity Colors
            overlay_frame = self._create_overlay(frame, merged_poses, phones, events)
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Keep only recent processing times for averaging
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            logger.debug(f"Frame {self.frame_count} processed in {processing_time:.3f}s")
            
            return overlay_frame, events
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {e}")
            # Return original frame and empty events on error
            return frame, []
    
    def _merge_tracking_and_poses(self, tracked_persons: List[Dict], 
                                 pose_estimates: List[Dict]) -> List[Dict]:
        """Merge tracking IDs with pose estimates using bbox matching."""
        merged_poses = []
        
        for tracked_person in tracked_persons:
            track_id = tracked_person['track_id']
            track_bbox = tracked_person['bbox']
            
            # Find best matching pose estimate
            best_match = None
            best_iou = 0.0
            
            for pose in pose_estimates:
                pose_bbox = pose['bbox']
                iou = self._calculate_iou(track_bbox, pose_bbox)
                
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                    best_iou = iou
                    best_match = pose
            
            if best_match:
                # Merge tracking info with pose data
                merged_pose = best_match.copy()
                merged_pose['track_id'] = track_id
                merged_pose['bbox'] = track_bbox  # Use tracker bbox for consistency
                merged_poses.append(merged_pose)
            else:
                # Create default pose data for tracked person without pose match
                default_pose = {
                    'track_id': track_id,
                    'bbox': track_bbox,
                    'yaw': 0.0,
                    'pitch': 0.0,
                    'lean_flag': False,
                    'look_flag': False,
                    'phone_flag': False,
                    'confidence': tracked_person['conf']
                }
                merged_poses.append(default_pose)
        
        return merged_poses
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            if union <= 0:
                return 0.0
            
            return intersection / union
        
        except Exception:
            return 0.0
    
    def _create_overlay(self, frame: np.ndarray, merged_poses: List[Dict], 
                       phones: List[Dict], events: List[Dict]) -> np.ndarray:
        """Create visualization overlay with bounding boxes, labels, and severity colors."""
        overlay_frame = frame.copy()
        
        # Get active violations for color coding
        active_violations = get_active_violations()
        
        # Draw person bounding boxes with severity colors
        for pose in merged_poses:
            x1, y1, x2, y2 = [int(coord) for coord in pose['bbox']]
            track_id = pose['track_id']
            
            # Determine color based on severity
            color = (0, 255, 0)  # Default green (normal)
            status = "Normal"
            
            if track_id in active_violations:
                violation = active_violations[track_id]
                severity_color = violation.severity.color
                
                if severity_color == 'red':
                    color = (0, 0, 255)  # Red for cheating
                elif severity_color == 'orange':
                    color = (0, 165, 255)  # Orange for phone use
                elif severity_color == 'yellow':
                    color = (0, 255, 255)  # Yellow for minor violations
                
                status = violation.label
            
            # Draw bounding box
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 3)
            
            # Create label with track ID and status
            label = f"{track_id}: {status}"
            
            # Add behavior indicators
            indicators = []
            if pose['lean_flag']:
                indicators.append("L")
            if pose['look_flag']:
                indicators.append("H")
            if pose['phone_flag']:
                indicators.append("P")
            
            if indicators:
                label += f" [{','.join(indicators)}]"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(overlay_frame, (x1, y1-35), (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(overlay_frame, label, (x1+5, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw head direction indicator if significant
            if abs(pose.get('yaw', 0)) > 10:
                center_x, center_y = int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.2)
                arrow_length = 40
                yaw_rad = np.radians(pose['yaw'])
                arrow_x = center_x + int(arrow_length * np.sin(yaw_rad))
                arrow_y = center_y
                cv2.arrowedLine(overlay_frame, (center_x, center_y), (arrow_x, arrow_y), color, 2)
        
        # Draw phone detections
        for phone in phones:
            x1, y1, x2, y2 = [int(coord) for coord in phone['bbox']]
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta for phones
            cv2.putText(overlay_frame, "Phone", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Add system status overlay
        self._add_system_status_overlay(overlay_frame, events)
        
        return overlay_frame
    
    def _add_system_status_overlay(self, frame: np.ndarray, events: List[Dict]):
        """Add system status information to the frame."""
        height, width = frame.shape[:2]
        
        # System status
        active_violations = get_active_violations()
        cheating_count = sum(1 for v in active_violations.values() if v.severity == Severity.CHEATING)
        
        status_lines = [
            f"CheatGPT3 - Frame {self.frame_count}",
            f"Active Tracks: {len(self.tracker.get_active_tracks())}",
            f"Violations: {len(active_violations)}",
            f"Cheating: {cheating_count}"
        ]
        
        # Add timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_lines.append(f"Time: {current_time}")
        
        # Performance info
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            status_lines.append(f"FPS: {fps:.1f}")
        
        # Draw status background
        status_height = len(status_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (300, status_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, status_height), (255, 255, 255), 2)
        
        # Draw status text
        for i, line in enumerate(status_lines):
            color = (0, 0, 255) if "Cheating:" in line and cheating_count > 0 else (255, 255, 255)
            cv2.putText(frame, line, (15, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _save_evidence_frame(self, frame: np.ndarray, cheating_events: List[Dict], 
                            cam_id: str, timestamp: float):
        """Save evidence frame when cheating is detected."""
        try:
            # Create timestamp string for filename
            dt = datetime.fromtimestamp(timestamp)
            timestamp_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            
            # Create filename with event details
            track_ids = [event['track_id'] for event in cheating_events]
            track_str = "_".join(track_ids)
            filename = f"cheating_{cam_id}_{timestamp_str}_{track_str}.jpg"
            filepath = os.path.join(self.evidence_dir, filename)
            
            # Save frame
            success = cv2.imwrite(filepath, frame)
            
            if success:
                logger.warning(f"EVIDENCE SAVED: {filename}")
                logger.warning(f"Cheating detected for tracks: {track_ids}")
                
                # Store evidence info in database
                for event in cheating_events:
                    self.db.store_event(
                        timestamp=timestamp,
                        cam_id=cam_id,
                        track_id=event['track_id'],
                        event_type=event['event_type'],
                        confidence=event['confidence'],
                        bbox=event['bbox'],
                        evidence_path=filepath
                    )
            else:
                logger.error(f"Failed to save evidence frame: {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving evidence frame: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        tracker_stats = self.tracker.get_statistics()
        active_violations = get_active_violations()
        
        # Violation distribution
        severity_counts = {}
        for violation in active_violations.values():
            severity = violation.severity.label
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Performance metrics
        avg_processing_time = 0.0
        fps = 0.0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
        
        return {
            'frame_count': self.frame_count,
            'active_tracks': tracker_stats['active_tracks'],
            'total_tracks_created': tracker_stats['total_tracks_created'],
            'active_violations': len(active_violations),
            'severity_distribution': severity_counts,
            'performance': {
                'avg_processing_time': avg_processing_time,
                'fps': fps,
                'total_processing_samples': len(self.processing_times)
            },
            'evidence_directory': self.evidence_dir
        }
    
    def reset(self):
        """Reset engine state (useful for testing or new sessions)."""
        logger.info("Resetting engine state...")
        self.frame_count = 0
        self.processing_times = []
        self.tracker = Tracker(max_disappeared=30, max_distance=100.0)
        logger.info("Engine state reset complete")
    
    def __repr__(self):
        """String representation of the engine."""
        stats = self.get_statistics()
        return (f"Engine(frames={stats['frame_count']}, "
                f"tracks={stats['active_tracks']}, "
                f"violations={stats['active_violations']}, "
                f"fps={stats['performance']['fps']:.1f})")

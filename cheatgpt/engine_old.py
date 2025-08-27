"""Real-time CheatGPT3 Engine with YOLOv11 + Pose Rules + LSTM.

This engine processes webcam frames in real-time using:
- YOLOv11 for person and phone detection (GPU-accelerated)
- Pose analysis for behavior extraction
- Rule-based immediate evaluation
- LSTM for temporal pattern recognition
- GPU-first with CPU fallback
"""

import os
import time
import logging
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from .detectors.yolo11_detector import YOLO11Detector
from .detectors.pose_detector import PoseDetector
from .db.db_manager import DBManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehaviorLSTM(nn.Module):
    """LSTM network for temporal behavior pattern recognition."""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=4, sequence_length=30):
        super(BehaviorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)  # Normal, Suspicious, Phone, Cheating
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        """Forward pass through LSTM."""
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classify
        output = self.classifier(last_output)
        return output

class Engine:
    """Real-time CheatGPT3 Engine with GPU acceleration."""
    
    def __init__(self):
        """Initialize the real-time engine with GPU-first configuration."""
        logger.info("🚀 Initializing Real-time CheatGPT3 Engine...")
        
        # Device configuration (GPU-first with CPU fallback)
        self.device = self._setup_device()
        logger.info(f"🔧 Using device: {self.device}")
        
        # Detection thresholds
        self.person_conf_thresh = float(os.getenv('PERSON_CONF_THRESH', '0.4'))
        self.phone_conf_thresh = float(os.getenv('PHONE_CONF_THRESH', '0.4'))
        self.lean_angle_thresh = float(os.getenv('LEAN_ANGLE_THRESH', '12.0'))
        self.head_turn_thresh = float(os.getenv('HEAD_TURN_THRESH', '15.0'))
        
        # Initialize components
        self._initialize_components()
        
        # LSTM configuration
        self.sequence_length = 30  # 30 frames for temporal analysis
        self.behavior_history = []  # Store behavior sequences
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.gpu_memory_usage = []
        
        # Evidence storage
        self.evidence_dir = "uploads/evidence"
        self._ensure_evidence_directory()
        
        logger.info("✅ Real-time Engine initialization complete")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device with GPU-first priority."""
        force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        
        if force_cpu:
            device = torch.device('cpu')
            logger.info("🖥️  Forced CPU mode")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Optimize GPU settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            device = torch.device('cpu')
            logger.info("🖥️  No GPU available, using CPU")
        
        return device
    
    def _initialize_components(self):
        """Initialize all detection and analysis components."""
        logger.info("🔧 Initializing components...")
        
        # YOLOv11 Detector (GPU-accelerated)
        self.yolo_detector = YOLO11Detector()
        logger.info("✅ YOLOv11 detector ready")
        
        # Pose Detector
        self.pose_detector = PoseDetector()
        logger.info("✅ Pose detector ready")
        
        # LSTM Model for temporal analysis
        self.lstm_model = BehaviorLSTM(
            input_size=6,  # lean_flag, look_flag, phone_flag, lean_angle, head_angle, confidence
            hidden_size=64,
            num_layers=2,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        self.lstm_model.eval()  # Set to evaluation mode
        logger.info(f"✅ LSTM model ready on {self.device}")
        
        # Database manager
        self.db = DBManager()
        logger.info("✅ Database manager ready")
    
    def _ensure_evidence_directory(self):
        """Ensure evidence directory exists."""
        try:
            os.makedirs(self.evidence_dir, exist_ok=True)
            logger.info(f"📁 Evidence directory: {self.evidence_dir}")
        except Exception as e:
            logger.error(f"Failed to create evidence directory: {e}")
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
            
            # Step 2: Assign simple IDs based on position (no tracking)
            logger.debug(f"Frame {self.frame_count}: Assigning detection IDs...")
            detection_persons = self._assign_detection_ids(persons)
            
            logger.debug(f"Assigned IDs to {len(detection_persons)} persons")
            
            # Step 3: Pose Detector Extracts Behavior Features
            logger.debug(f"Frame {self.frame_count}: Analyzing poses...")
            pose_estimates = self.pose_detector.estimate(frame, phones)
            
            # Merge detection IDs with pose estimates
            merged_poses = self._merge_detections_and_poses(detection_persons, pose_estimates)
            
            logger.debug(f"Pose analysis complete for {len(merged_poses)} persons")
            
            # Step 4: Policy Module Applies Rules (frame-based, no history)
            logger.debug(f"Frame {self.frame_count}: Applying immediate policy rules...")
            events = []
            
            for pose in merged_poses:
                if self.debug_mode:
                    logger.info(f"Evaluating behavior for {pose['detection_id']}: "
                               f"lean={pose['lean_flag']}, look={pose['look_flag']}, phone={pose['phone_flag']}")
                
                # Apply immediate rule-based evaluation (no temporal tracking)
                violation_events = self._evaluate_immediate_behavior(
                    pose['detection_id'],
                    pose['lean_flag'],
                    pose['look_flag'],
                    pose['phone_flag'],
                    ts
                )
                
                # Convert violations to events
                for violation in violation_events:
                    event = {
                        'timestamp': ts,
                        'cam_id': cam_id,
                        'track_id': violation['detection_id'],
                        'event_type': violation['event_type'],
                        'severity': violation['severity'],
                        'confidence': violation['confidence'],
                        'bbox': pose['bbox'],
                        'details': violation['details']
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
        
        # Create event lookup for current frame
        event_lookup = {}
        for event in events:
            event_lookup[event['track_id']] = event
        
        # Draw person bounding boxes with severity colors
        for pose in merged_poses:
            x1, y1, x2, y2 = [int(coord) for coord in pose['bbox']]
            detection_id = pose['detection_id']
            
            # Determine color based on current frame events
            color = (0, 255, 0)  # Default green (normal)
            status = "Normal"
            
            if detection_id in event_lookup:
                event = event_lookup[detection_id]
                severity = event['severity']
                
                if severity == 'red':
                    color = (0, 0, 255)  # Red for cheating
                elif severity == 'orange':
                    color = (0, 165, 255)  # Orange for high suspicion
                elif severity == 'yellow':
                    color = (0, 255, 255)  # Yellow for minor violations
                
                status = event['event_type']
            
            # Draw bounding box
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 3)
            
            # Create label with detection ID and status
            label = f"{detection_id}: {status}"
            
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
        self._add_system_status_overlay(overlay_frame, merged_poses, events)
        
        return overlay_frame
    
    def _add_system_status_overlay(self, frame: np.ndarray, merged_poses: List[Dict], events: List[Dict]):
        """Add system status information to the frame."""
        height, width = frame.shape[:2]
        
        status_lines = [
            f"CheatGPT3 - Frame {self.frame_count}",
            f"Active Detections: {len(merged_poses)}",
            f"Current Events: {len(events)}",
            f"Violations: {len([e for e in events if e['severity'] in ['orange', 'red']])}"
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
            # Highlight violations in red
            violation_count = len([e for e in events if e['severity'] in ['orange', 'red']])
            color = (0, 0, 255) if "Violations:" in line and violation_count > 0 else (255, 255, 255)
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
        """Get comprehensive engine statistics (without tracking)."""
        # Performance metrics
        avg_processing_time = 0.0
        fps = 0.0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
        
        return {
            'frame_count': self.frame_count,
            'active_tracks': 0,  # No tracking
            'total_tracks_created': 0,  # No tracking
            'active_violations': 0,  # Frame-based, no persistent violations
            'severity_distribution': {},  # Frame-based
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
        # No tracker to reset
        logger.info("Engine state reset complete")
    
    def _assign_detection_ids(self, persons: List[Dict]) -> List[Dict]:
        """Assign simple IDs to detected persons based on their position."""
        # Sort persons by x-coordinate (left to right)
        sorted_persons = sorted(persons, key=lambda p: p['bbox'][0])
        
        detection_persons = []
        for i, person in enumerate(sorted_persons):
            detection_person = person.copy()
            detection_person['detection_id'] = f"person_{i:03d}"
            detection_persons.append(detection_person)
        
        return detection_persons
    
    def _merge_detections_and_poses(self, detection_persons: List[Dict], 
                                  pose_estimates: List[Dict]) -> List[Dict]:
        """Merge detection IDs with pose estimates using spatial matching."""
        merged_poses = []
        
        for detection in detection_persons:
            detection_bbox = detection['bbox']
            detection_center = self._get_bbox_center(detection_bbox)
            
            # Find closest pose estimate
            best_pose = None
            min_distance = float('inf')
            
            for pose in pose_estimates:
                pose_bbox = pose['bbox']
                pose_center = self._get_bbox_center(pose_bbox)
                
                # Calculate distance between centers
                distance = self._calculate_distance(detection_center, pose_center)
                
                if distance < min_distance:
                    min_distance = distance
                    best_pose = pose
            
            # Merge detection and pose data
            if best_pose and min_distance < 100:  # Reasonable distance threshold
                merged_pose = best_pose.copy()
                merged_pose['detection_id'] = detection['detection_id']
                merged_pose['bbox'] = detection_bbox  # Use detection bbox
                merged_pose['confidence'] = detection['conf']
                merged_poses.append(merged_pose)
            else:
                # No matching pose, create default
                default_pose = {
                    'detection_id': detection['detection_id'],
                    'bbox': detection_bbox,
                    'confidence': detection['conf'],
                    'lean_flag': False,
                    'look_flag': False,
                    'phone_flag': False,
                    'head_direction': 'forward',
                    'lean_angle': 0.0,
                    'head_turn_angle': 0.0
                }
                merged_poses.append(default_pose)
        
        return merged_poses
    
    def _evaluate_immediate_behavior(self, detection_id: str, lean_flag: bool, 
                                   look_flag: bool, phone_flag: bool, 
                                   timestamp: float) -> List[Dict]:
        """Evaluate behavior immediately without temporal tracking."""
        violations = []
        
        # Immediate rule-based evaluation
        if phone_flag:
            violations.append({
                'detection_id': detection_id,
                'event_type': 'Phone Use Detected',
                'severity': 'red',  # Immediate cheating
                'confidence': 0.9,
                'details': 'Phone detected in hand - potential cheating'
            })
        elif lean_flag and look_flag:
            violations.append({
                'detection_id': detection_id,
                'event_type': 'Multiple Suspicious Behaviors',
                'severity': 'orange',  # High suspicion
                'confidence': 0.8,
                'details': 'Leaning and looking around simultaneously'
            })
        elif lean_flag:
            violations.append({
                'detection_id': detection_id,
                'event_type': 'Suspicious Posture',
                'severity': 'yellow',  # Minor violation
                'confidence': 0.6,
                'details': 'Leaning detected - monitor closely'
            })
        elif look_flag:
            violations.append({
                'detection_id': detection_id,
                'event_type': 'Looking Around',
                'severity': 'yellow',  # Minor violation
                'confidence': 0.6,
                'details': 'Head turning detected - possible distraction'
            })
        
        return violations
    
    def _get_bbox_center(self, bbox: List[float]) -> tuple:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, point1: tuple, point2: tuple) -> float:
        """Calculate Euclidean distance between two points."""
        import math
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def __repr__(self):
        """String representation of the engine."""
        stats = self.get_statistics()
        return (f"Engine(frames={stats['frame_count']}, "
                f"tracks={stats['active_tracks']}, "
                f"violations={stats['active_violations']}, "
                f"fps={stats['performance']['fps']:.1f})")

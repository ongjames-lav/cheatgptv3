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
    
    def process_frame(self, frame: np.ndarray, cam_id: str = "webcam", 
                     ts: Optional[float] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame in real-time with GPU acceleration.
        
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
        
        try:
            # Step 1: GPU-accelerated YOLOv11 Detection
            detections = self._detect_objects(frame)
            persons = [d for d in detections if d['cls_name'] == 'person']
            phones = [d for d in detections if d['cls_name'] == 'cell phone']
            
            logger.debug(f"Frame {self.frame_count}: {len(persons)} persons, {len(phones)} phones")
            
            # Step 2: Pose Analysis
            pose_results = self._analyze_poses(frame, persons, phones)
            
            # Step 3: Rule-based Immediate Evaluation
            immediate_events = self._evaluate_immediate_rules(pose_results, ts)
            
            # Step 4: LSTM Temporal Analysis
            temporal_events = self._analyze_temporal_patterns(pose_results, ts)
            
            # Combine events
            all_events = immediate_events + temporal_events
            
            # Step 5: Save Evidence for Critical Events
            critical_events = [e for e in all_events if e['severity'] in ['red', 'critical']]
            if critical_events:
                self._save_evidence_frame(frame, critical_events, cam_id, ts)
            
            # Step 6: Create Visualization Overlay
            overlay_frame = self._create_real_time_overlay(frame, pose_results, phones, all_events)
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # GPU memory tracking
            if self.device.type == 'cuda':
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                self.gpu_memory_usage.append(gpu_memory)
            
            # Keep recent performance data
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
                if self.gpu_memory_usage:
                    self.gpu_memory_usage.pop(0)
            
            return overlay_frame, all_events
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {e}")
            return frame, []
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """GPU-accelerated object detection with YOLOv11."""
        try:
            # Run YOLOv11 detection
            detections = self.yolo_detector.detect(frame)
            
            # Filter by confidence thresholds
            filtered_detections = []
            for det in detections:
                min_conf = self.person_conf_thresh if det['cls_name'] == 'person' else self.phone_conf_thresh
                if det['conf'] >= min_conf:
                    filtered_detections.append(det)
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _analyze_poses(self, frame: np.ndarray, persons: List[Dict], 
                      phones: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze poses for each detected person."""
        try:
            # Get pose estimates
            pose_estimates = self.pose_detector.estimate(frame, phones)
            
            # Match poses with person detections
            matched_poses = []
            for i, person in enumerate(persons):
                person_center = self._get_bbox_center(person['bbox'])
                
                # Find closest pose
                best_pose = None
                min_distance = float('inf')
                
                for pose in pose_estimates:
                    pose_center = self._get_bbox_center(pose['bbox'])
                    distance = self._calculate_distance(person_center, pose_center)
                    
                    if distance < min_distance and distance < 100:
                        min_distance = distance
                        best_pose = pose
                
                # Create combined result
                if best_pose:
                    result = best_pose.copy()
                    result['person_id'] = f"person_{i:03d}"
                    result['bbox'] = person['bbox']  # Use person detection bbox
                    result['detection_conf'] = person['conf']
                else:
                    # Default pose for person without pose detection
                    result = {
                        'person_id': f"person_{i:03d}",
                        'bbox': person['bbox'],
                        'detection_conf': person['conf'],
                        'lean_flag': False,
                        'look_flag': False,
                        'phone_flag': False,
                        'lean_angle': 0.0,
                        'head_turn_angle': 0.0,
                        'confidence': 0.5
                    }
                
                matched_poses.append(result)
            
            return matched_poses
            
        except Exception as e:
            logger.error(f"Pose analysis error: {e}")
            return []
    
    def _evaluate_immediate_rules(self, pose_results: List[Dict], 
                                 timestamp: float) -> List[Dict[str, Any]]:
        """Apply immediate rule-based evaluation."""
        events = []
        
        for pose in pose_results:
            person_id = pose['person_id']
            
            # Immediate rule evaluation
            if pose['phone_flag']:
                events.append({
                    'timestamp': timestamp,
                    'person_id': person_id,
                    'event_type': 'Phone Use Detected',
                    'severity': 'red',
                    'confidence': 0.95,
                    'source': 'immediate_rules',
                    'details': 'Phone detected - immediate cheating alert',
                    'bbox': pose['bbox']
                })
            elif pose['lean_flag'] and pose['look_flag']:
                events.append({
                    'timestamp': timestamp,
                    'person_id': person_id,
                    'event_type': 'Multiple Suspicious Behaviors',
                    'severity': 'orange',
                    'confidence': 0.8,
                    'source': 'immediate_rules',
                    'details': f"Leaning ({pose['lean_angle']:.1f}°) + Looking around ({pose['head_turn_angle']:.1f}°)",
                    'bbox': pose['bbox']
                })
            elif pose['lean_flag']:
                events.append({
                    'timestamp': timestamp,
                    'person_id': person_id,
                    'event_type': 'Suspicious Posture',
                    'severity': 'yellow',
                    'confidence': 0.6,
                    'source': 'immediate_rules',
                    'details': f"Leaning detected ({pose['lean_angle']:.1f}°)",
                    'bbox': pose['bbox']
                })
            elif pose['look_flag']:
                events.append({
                    'timestamp': timestamp,
                    'person_id': person_id,
                    'event_type': 'Looking Around',
                    'severity': 'yellow',
                    'confidence': 0.6,
                    'source': 'immediate_rules',
                    'details': f"Head turning detected ({pose['head_turn_angle']:.1f}°)",
                    'bbox': pose['bbox']
                })
        
        return events
    
    def _analyze_temporal_patterns(self, pose_results: List[Dict], 
                                  timestamp: float) -> List[Dict[str, Any]]:
        """Analyze temporal patterns using LSTM."""
        if not pose_results:
            return []
        
        events = []
        
        try:
            # Update behavior history
            current_behaviors = []
            for pose in pose_results:
                behavior_vector = [
                    float(pose['lean_flag']),
                    float(pose['look_flag']),
                    float(pose['phone_flag']),
                    pose['lean_angle'] / 90.0,  # Normalize to [0,1]
                    pose['head_turn_angle'] / 180.0,  # Normalize to [0,1]
                    pose['confidence']
                ]
                current_behaviors.append(behavior_vector)
            
            # Average behaviors across all detected persons
            if current_behaviors:
                avg_behavior = np.mean(current_behaviors, axis=0)
                self.behavior_history.append(avg_behavior)
            
            # Keep only recent history
            if len(self.behavior_history) > self.sequence_length:
                self.behavior_history.pop(0)
            
            # LSTM analysis if we have enough history
            if len(self.behavior_history) >= self.sequence_length:
                sequence = torch.tensor(
                    [self.behavior_history[-self.sequence_length:]], 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                with torch.no_grad():
                    lstm_output = self.lstm_model(sequence)
                    probabilities = torch.softmax(lstm_output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                # Interpret LSTM predictions
                class_labels = ['Normal', 'Suspicious', 'Phone Use', 'Cheating Pattern']
                
                if predicted_class > 0 and confidence > 0.7:  # Threshold for temporal events
                    for pose in pose_results:
                        events.append({
                            'timestamp': timestamp,
                            'person_id': pose['person_id'],
                            'event_type': f'Temporal {class_labels[predicted_class]}',
                            'severity': 'orange' if predicted_class == 1 else 'red',
                            'confidence': confidence,
                            'source': 'lstm_temporal',
                            'details': f'LSTM detected {class_labels[predicted_class].lower()} pattern over {self.sequence_length} frames',
                            'bbox': pose['bbox']
                        })
        
        except Exception as e:
            logger.error(f"LSTM temporal analysis error: {e}")
        
        return events
    
    def _create_real_time_overlay(self, frame: np.ndarray, pose_results: List[Dict], 
                                 phones: List[Dict], events: List[Dict]) -> np.ndarray:
        """Create real-time visualization overlay."""
        overlay_frame = frame.copy()
        
        # Create event lookup
        event_lookup = {}
        for event in events:
            person_id = event['person_id']
            if person_id not in event_lookup or event['severity'] == 'red':
                event_lookup[person_id] = event
        
        # Draw person bounding boxes
        for pose in pose_results:
            x1, y1, x2, y2 = [int(coord) for coord in pose['bbox']]
            person_id = pose['person_id']
            
            # Determine color based on events
            color = (0, 255, 0)  # Default green
            status = "Normal"
            
            if person_id in event_lookup:
                event = event_lookup[person_id]
                severity = event['severity']
                
                if severity == 'red':
                    color = (0, 0, 255)  # Red
                elif severity == 'orange':
                    color = (0, 165, 255)  # Orange
                elif severity == 'yellow':
                    color = (0, 255, 255)  # Yellow
                
                status = event['event_type']
            
            # Draw bounding box
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{person_id}: {status}"
            
            # Add behavior indicators
            indicators = []
            if pose['lean_flag']:
                indicators.append(f"L{pose['lean_angle']:.0f}°")
            if pose['look_flag']:
                indicators.append(f"H{pose['head_turn_angle']:.0f}°")
            if pose['phone_flag']:
                indicators.append("📱")
            
            if indicators:
                label += f" [{' '.join(indicators)}]"
            
            # Draw label background and text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(overlay_frame, (x1, y1-35), (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(overlay_frame, label, (x1+5, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw phone detections
        for phone in phones:
            x1, y1, x2, y2 = [int(coord) for coord in phone['bbox']]
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(overlay_frame, f"Phone {phone['conf']:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add system status
        self._add_system_status(overlay_frame, pose_results, events)
        
        return overlay_frame
    
    def _add_system_status(self, frame: np.ndarray, pose_results: List[Dict], events: List[Dict]):
        """Add system status overlay."""
        # Performance metrics
        avg_fps = 1.0 / np.mean(self.processing_times[-30:]) if self.processing_times else 0
        gpu_memory = np.mean(self.gpu_memory_usage[-10:]) if self.gpu_memory_usage else 0
        
        # Status lines
        status_lines = [
            f"CheatGPT3 Real-time | Frame: {self.frame_count}",
            f"FPS: {avg_fps:.1f} | Device: {self.device}",
            f"Persons: {len(pose_results)} | Events: {len(events)}",
            f"GPU Mem: {gpu_memory:.1f}GB" if self.device.type == 'cuda' else "CPU Mode"
        ]
        
        # Draw status background
        status_height = len(status_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (350, status_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, status_height), (255, 255, 255), 2)
        
        # Draw status text
        for i, line in enumerate(status_lines):
            color = (0, 255, 0) if "Events: 0" in line else (255, 255, 255)
            if len(events) > 0:
                color = (0, 0, 255)  # Red if events detected
            cv2.putText(frame, line, (15, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _save_evidence_frame(self, frame: np.ndarray, events: List[Dict], 
                            cam_id: str, timestamp: float):
        """Save evidence frame for critical events."""
        try:
            dt = datetime.fromtimestamp(timestamp)
            timestamp_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            person_ids = [event['person_id'] for event in events]
            person_str = "_".join(person_ids)
            filename = f"critical_{cam_id}_{timestamp_str}_{person_str}.jpg"
            filepath = os.path.join(self.evidence_dir, filename)
            
            cv2.imwrite(filepath, frame)
            logger.warning(f"🚨 EVIDENCE SAVED: {filename}")
            
            # Store in database
            for event in events:
                self.db.store_event(
                    timestamp=timestamp,
                    cam_id=cam_id,
                    track_id=event['person_id'],
                    event_type=event['event_type'],
                    confidence=event['confidence'],
                    evidence_path=filepath
                )
        
        except Exception as e:
            logger.error(f"Failed to save evidence: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive real-time engine statistics."""
        avg_fps = 1.0 / np.mean(self.processing_times[-30:]) if self.processing_times else 0
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_gpu_memory = np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0
        
        return {
            'frame_count': self.frame_count,
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available(),
            'performance': {
                'avg_fps': avg_fps,
                'avg_processing_time_ms': avg_processing_time * 1000,
                'avg_gpu_memory_gb': avg_gpu_memory,
                'total_processing_samples': len(self.processing_times)
            },
            'lstm': {
                'sequence_length': self.sequence_length,
                'behavior_history_length': len(self.behavior_history),
                'model_device': str(next(self.lstm_model.parameters()).device)
            },
            'thresholds': {
                'person_confidence': self.person_conf_thresh,
                'phone_confidence': self.phone_conf_thresh,
                'lean_angle': self.lean_angle_thresh,
                'head_turn': self.head_turn_thresh
            },
            'evidence_directory': self.evidence_dir,
            # Legacy compatibility
            'total_tracks_created': 0,
            'active_violations': 0,
            'active_tracks': 0
        }
    
    def reset(self):
        """Reset engine state."""
        logger.info("🔄 Resetting real-time engine state...")
        self.frame_count = 0
        self.processing_times.clear()
        self.gpu_memory_usage.clear()
        self.behavior_history.clear()
        
        # Clear GPU cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info("✅ Engine state reset complete")
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def __repr__(self):
        """String representation of the engine."""
        stats = self.get_statistics()
        return (f"Engine(device={stats['device']}, "
                f"frames={stats['frame_count']}, "
                f"fps={stats['performance']['avg_fps']:.1f})")

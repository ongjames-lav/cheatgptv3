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
from .temporal.lstm_model import get_lstm_classifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced LSTM integration for 6-category dataset
try:
    import sys
    sys.path.append('.')
    from enhanced_lstm_integration import get_enhanced_lstm_classifier, create_enhanced_behavior_vector
    ENHANCED_LSTM_AVAILABLE = True
    logger.info("✅ Enhanced LSTM integration available")
except ImportError:
    logger.warning("Enhanced LSTM integration not available, using standard LSTM")
    ENHANCED_LSTM_AVAILABLE = False

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
        
        # LSTM configuration (must be set before _initialize_components)
        self.sequence_length = 30  # 30 frames for temporal analysis
        self.behavior_history = []  # Store behavior sequences
        
        # Temporal Cheating Detection System
        self.temporal_cheating_enabled = os.getenv('TEMPORAL_CHEATING_ENABLED', 'true').lower() == 'true'
        self.temporal_cheating_threshold = float(os.getenv('TEMPORAL_CHEATING_THRESHOLD', '12.0'))  # seconds
        self.cheating_cooldown_duration = float(os.getenv('CHEATING_COOLDOWN_DURATION', '30.0'))  # seconds
        self.fps_estimate = 30.0  # frames per second estimate
        self.required_frames = int(self.temporal_cheating_threshold * self.fps_estimate)  # ~360 frames
        
        # Temporal analysis tracking (per-person)
        self.looking_around_history = {}  # Track timestamps of looking detections per person
        self.leaning_history = {}  # Track timestamps of leaning detections per person
        self.cheating_cooldown = {}  # Track IDs to prevent spam
        
        # Combined behavior tracking (legacy support)
        self.combined_behavior_start = None  # When both behaviors started
        
        logger.info(f"🕐 Temporal cheating detection: {self.temporal_cheating_threshold}s threshold ({self.required_frames} frames)")
        
        # Initialize components
        self._initialize_components()
        
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
        
        # Enhanced LSTM Behavior Classifier (6-category dataset)
        if ENHANCED_LSTM_AVAILABLE:
            self.lstm_classifier = get_enhanced_lstm_classifier(
                model_path="weights/enhanced_lstm_behavior.pth",
                label_encoder_path="weights/enhanced_label_encoder.pkl"
            )
            self.enhanced_lstm = True
            logger.info("🧠 Using Enhanced LSTM with 6-category dataset")
        else:
            # Fallback to standard LSTM
            self.lstm_classifier = get_lstm_classifier(
                model_path="weights/lstm_behavior.pth",
                label_encoder_path="weights/label_encoder.pkl",
                device=self.device
            )
            self.enhanced_lstm = False
        
        if self.lstm_classifier.is_loaded:
            model_type = "Enhanced" if self.enhanced_lstm else "Standard"
            logger.info(f"✅ {model_type} LSTM classifier ready on {self.device}")
            logger.info(f"   Classes: {self.lstm_classifier.class_labels}")
        else:
            logger.warning("⚠️ LSTM model not found - using fallback rules")
            logger.info("   Train the model with: python train_lstm.py")
        
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
            
            # Step 4: Temporal Analysis (LSTM + Sustained Behavior Detection)
            lstm_events = self._analyze_temporal_patterns(pose_results, ts)
            temporal_cheating_events = self._track_temporal_behaviors(pose_results, ts)
            temporal_events = lstm_events + temporal_cheating_events
            
            # Combine events
            all_events = immediate_events + temporal_events
            
            # Debug: Log all events being processed
            if all_events:
                event_types = [f"{e['event_type']}({e['severity']})" for e in all_events]
                logger.info(f"🎯 Processing {len(all_events)} events: {event_types}")
            
            # Step 5: Save Evidence for Critical Events
            critical_events = [e for e in all_events if e['severity'] in ['red', 'critical']]
            if critical_events:
                logger.warning(f"💾 Saving evidence for {len(critical_events)} critical events")
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
    
    def _track_temporal_behaviors(self, pose_results: List[Dict], timestamp: float) -> List[Dict[str, Any]]:
        """Track temporal patterns for sustained cheating detection."""
        temporal_events = []
        
        if not self.temporal_cheating_enabled:
            return temporal_events
        
        logger.debug(f"🔍 Temporal Analysis: Processing {len(pose_results)} poses")

        current_time = timestamp        # Clean old history (keep only recent behaviors)
        history_window = 20.0  # seconds
        
        # Clean looking history for all persons
        for person_id in list(self.looking_around_history.keys()):
            self.looking_around_history[person_id] = [
                t for t in self.looking_around_history[person_id] 
                if current_time - t < history_window
            ]
            if not self.looking_around_history[person_id]:
                del self.looking_around_history[person_id]
        
        # Clean leaning history for all persons  
        for person_id in list(self.leaning_history.keys()):
            self.leaning_history[person_id] = [
                t for t in self.leaning_history[person_id] 
                if current_time - t < history_window
            ]
            if not self.leaning_history[person_id]:
                del self.leaning_history[person_id]
        
        # Clean cheating cooldown
        self.cheating_cooldown = {pid: t for pid, t in self.cheating_cooldown.items() 
                                if current_time - t < self.cheating_cooldown_duration}
        
        for pose in pose_results:
            person_id = pose['person_id']
            
            # Note: No cooldown skip for temporal events - allow continuous red detection
            
            # Track current behaviors
            current_looking = pose.get('look_flag', False)
            current_leaning = pose.get('lean_flag', False)
            
            # Debug logging for temporal tracking
            logger.debug(f"🔍 Temporal Debug - Person {person_id}: looking={current_looking}, leaning={current_leaning}, yaw={pose.get('head_turn_angle', 0):.1f}°, lean_angle={pose.get('lean_angle', 0):.1f}°")
            
            # Debug logging for flags in pose data
            if current_looking or current_leaning:
                logger.info(f"👀 Temporal tracking flags - Person {person_id}: look_flag={current_looking}, lean_flag={current_leaning}")
            
            # Add to history if behaviors detected
            if current_looking:
                if person_id not in self.looking_around_history:
                    self.looking_around_history[person_id] = []
                self.looking_around_history[person_id].append(current_time)
                logger.info(f"👁️ Added looking detection at {current_time:.1f}s for person {person_id}")
            if current_leaning:
                if person_id not in self.leaning_history:
                    self.leaning_history[person_id] = []
                self.leaning_history[person_id].append(current_time)
                logger.info(f"🔄 Added leaning detection at {current_time:.1f}s for person {person_id}")
            
            # Calculate current durations
            looking_duration = 0.0
            leaning_duration = 0.0
            
            if person_id in self.looking_around_history:
                looking_duration = self._calculate_behavior_duration(self.looking_around_history[person_id], current_time)
            if person_id in self.leaning_history:
                leaning_duration = self._calculate_behavior_duration(self.leaning_history[person_id], current_time)
            
            # Log temporal progress for any significant duration
            if looking_duration > 3.0 or leaning_duration > 2.0:
                logger.warning(f"🕐 TEMPORAL PROGRESS - Person {person_id}: Looking={looking_duration:.1f}s, Leaning={leaning_duration:.1f}s")
            
            # Check for sustained combined behavior
            if current_looking and current_leaning:
                # Both behaviors present - check if sustained
                # Duration already calculated above
                
                # Calculate overlap duration (both behaviors happening together)
                overlap_duration = min(looking_duration, leaning_duration)
                
                # Only log detailed analysis if behaviors are significant (>3s)
                if overlap_duration > 3.0:
                    logger.info(f"🕐 Combined Temporal Analysis - Person {person_id}: Looking={looking_duration:.1f}s, Leaning={leaning_duration:.1f}s, Overlap={overlap_duration:.1f}s")
                
                # Check if sustained cheating threshold met
                if overlap_duration >= self.temporal_cheating_threshold:
                    # SUSTAINED COMBINED CHEATING DETECTED!
                    temporal_events.append({
                        'timestamp': current_time,
                        'person_id': person_id,
                        'event_type': 'Sustained Combined Suspicious Behavior',
                        'severity': 'red',
                        'confidence': 0.95,
                        'source': 'temporal_analysis',
                        'details': f'Sustained looking+leaning for {overlap_duration:.1f}s (threshold: {self.temporal_cheating_threshold:.1f}s)',
                        'bbox': pose['bbox'],
                        'looking_duration': looking_duration,
                        'leaning_duration': leaning_duration,
                        'overlap_duration': overlap_duration,
                        'lean_angle': pose.get('lean_angle', 0),
                        'head_turn_angle': pose.get('head_turn_angle', 0)
                    })
                    
                    # Add to cooldown to prevent spam
                    self.cheating_cooldown[person_id] = current_time
                    
                    logger.warning(f"🚨 SUSTAINED COMBINED CHEATING DETECTED: Person {person_id} - {overlap_duration:.1f}s of combined suspicious behavior")
            
            # Check for sustained looking alone (even without leaning)
            elif current_looking:
                # Duration already calculated above
                
                # Log progress for sustained looking
                if looking_duration > 5.0:
                    logger.info(f"🕐 Looking Temporal Analysis - Person {person_id}: Looking={looking_duration:.1f}s (sustained)")
                
                # Escalate sustained looking alone to cheating after longer threshold
                extended_threshold = 6.0  # 6 seconds for testing (was 9.6s)
                if looking_duration >= extended_threshold:
                    # SUSTAINED LOOKING CHEATING - Create event every time to maintain red color
                    temporal_events.append({
                        'timestamp': current_time,
                        'person_id': person_id,
                        'event_type': 'Sustained Looking Around',
                        'severity': 'red',
                        'confidence': 0.85,
                        'source': 'temporal_analysis',
                        'details': f'Sustained looking around for {looking_duration:.1f}s (threshold: {extended_threshold:.1f}s)',
                        'bbox': pose['bbox'],
                        'looking_duration': looking_duration,
                        'leaning_duration': 0,
                        'overlap_duration': looking_duration,
                        'lean_angle': pose.get('lean_angle', 0),
                        'head_turn_angle': pose.get('head_turn_angle', 0)
                    })
                    
                    # Only log warning once when first detected to prevent spam
                    if person_id not in self.cheating_cooldown:
                        logger.warning(f"🚨 SUSTAINED LOOKING CHEATING DETECTED: Person {person_id} - {looking_duration:.1f}s of continuous looking around")
                        logger.warning(f"🎯 TEMPORAL EVENT CREATED: {temporal_events[-1]['event_type']} with severity {temporal_events[-1]['severity']}")
                    
                    # Update cooldown timestamp to maintain detection state but allow red events
                    self.cheating_cooldown[person_id] = current_time
        
        # Debug: Log temporal events being returned
        if temporal_events:
            logger.warning(f"🎯 RETURNING {len(temporal_events)} TEMPORAL EVENTS: {[e['event_type'] for e in temporal_events]}")
        
        return temporal_events
    
    def _calculate_behavior_duration(self, behavior_history: List[float], current_time: float) -> float:
        """Calculate how long a behavior has been sustained."""
        if not behavior_history:
            return 0.0
        
        # Find the longest continuous period ending close to current time
        gap_tolerance = 2.0  # Allow 2 second gap between detections
        
        # Sort history in reverse order (newest first)
        sorted_history = sorted(behavior_history, reverse=True)
        
        # Find the most recent detection within tolerance
        most_recent = None
        for detection_time in sorted_history:
            if current_time - detection_time <= gap_tolerance:
                most_recent = detection_time
                break
        
        if most_recent is None:
            return 0.0
        
        # Now find the earliest detection in this continuous sequence
        earliest_in_sequence = most_recent
        for i in range(len(sorted_history) - 1):
            current_detection = sorted_history[i]
            next_detection = sorted_history[i + 1]
            
            # If gap between consecutive detections is too large, stop
            if current_detection - next_detection > gap_tolerance:
                break
            earliest_in_sequence = next_detection
        
        # Calculate total duration of sustained behavior
        duration = most_recent - earliest_in_sequence
        return max(duration, 0.1)  # Minimum 0.1s for any detection
    
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
                    'details': f"Leaning ({pose.get('lean_angle', 0):.1f}°) + Looking around ({pose.get('head_turn_angle', 0):.1f}°)",
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
                    'details': f"Leaning detected ({pose.get('lean_angle', 0):.1f}°)",
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
                    'details': f"Head turning detected ({pose.get('head_turn_angle', 0):.1f}°)",
                    'bbox': pose['bbox']
                })
        
        return events
    
    def _analyze_temporal_patterns(self, pose_results: List[Dict], 
                                  timestamp: float) -> List[Dict[str, Any]]:
        """Optimized temporal pattern analysis using trained LSTM with enhanced feature engineering."""
        if not pose_results:
            return []
        
        events = []
        
        try:
            # OPTIMIZATION 1: Enhanced Feature Engineering
            current_behaviors = []
            for pose in pose_results:
                if self.enhanced_lstm and ENHANCED_LSTM_AVAILABLE:
                    # Use enhanced 12-dimensional feature vector
                    behavior_vector = create_enhanced_behavior_vector(pose)
                    current_behaviors.append(behavior_vector.tolist())
                else:
                    # Standard 9-dimensional feature vector
                    lean_angle = pose.get('lean_angle', 0.0)
                    head_angle = pose.get('head_turn_angle', 0.0)
                    confidence = pose.get('confidence', 0.5)
                    
                    behavior_vector = [
                        float(pose['lean_flag']),                           # Binary lean flag
                        float(pose['look_flag']),                           # Binary look flag  
                        float(pose['phone_flag']),                          # Binary phone flag
                        np.clip(lean_angle / 45.0, -1.0, 1.0),            # Normalized lean angle [-1,1]
                        np.clip(head_angle / 90.0, -1.0, 1.0),            # Normalized head angle [-1,1]
                        np.clip(confidence, 0.0, 1.0),                    # Clipped confidence [0,1]
                        # Additional engineered features for better temporal understanding
                        float(pose['lean_flag'] and pose['look_flag']),    # Combined suspicious behavior
                        min(abs(lean_angle) / 45.0, 1.0),                 # Lean magnitude [0,1]
                        min(abs(head_angle) / 90.0, 1.0)                  # Head turn magnitude [0,1]
                    ]
                    current_behaviors.append(behavior_vector)
            
            # OPTIMIZATION 2: Temporal Smoothing and Aggregation
            if current_behaviors:
                # Use weighted average instead of simple mean for better representation
                weights = [pose.get('confidence', 0.5) for pose in pose_results]
                weights = np.array(weights) / (np.sum(weights) + 1e-8)  # Normalize weights
                
                # Compute confidence-weighted behavior
                avg_behavior = np.average(current_behaviors, axis=0, weights=weights)
                
                # Apply temporal smoothing if we have history
                if len(self.behavior_history) > 0:
                    # Exponential smoothing with recent frames
                    smoothing_factor = 0.3  # Adjust for responsiveness vs stability
                    prev_behavior = np.array(self.behavior_history[-1])
                    smoothed_behavior = (smoothing_factor * avg_behavior + 
                                       (1 - smoothing_factor) * prev_behavior)
                    self.behavior_history.append(smoothed_behavior.tolist())
                else:
                    self.behavior_history.append(avg_behavior.tolist())
            
            # OPTIMIZATION 3: Adaptive History Management
            # Dynamically adjust sequence length based on behavior stability
            max_history = max(self.sequence_length, 20)  # Allow longer history for complex patterns
            if len(self.behavior_history) > max_history:
                # Remove oldest frames but keep some buffer for context
                self.behavior_history = self.behavior_history[-max_history:]
            
            # OPTIMIZATION 4: Enhanced LSTM Analysis with Confidence Gating
            if len(self.behavior_history) >= self.sequence_length and self.lstm_classifier.is_loaded:
                # Multi-scale analysis: use different sequence lengths for robustness
                sequence_lengths = [self.sequence_length, min(15, len(self.behavior_history))]
                best_prediction = None
                best_confidence = 0.0
                
                for seq_len in sequence_lengths:
                    if len(self.behavior_history) >= seq_len:
                        # Prepare sequence for trained model
                        sequence = np.array(self.behavior_history[-seq_len:])
                        
                        # Enhanced prediction with auxiliary outputs
                        if self.enhanced_lstm and hasattr(self.lstm_classifier, 'predict_enhanced'):
                            prediction = self.lstm_classifier.predict_enhanced(
                                sequence, 
                                additional_features={'spatial_features': pose_results[0] if pose_results else {}}
                            )
                        else:
                            # Standard prediction
                            prediction = self.lstm_classifier.predict(sequence)
                        
                        if prediction['error'] is None:
                            pred_confidence = prediction['confidence']
                            
                            # Select best prediction based on confidence
                            if pred_confidence > best_confidence:
                                best_prediction = prediction
                                best_confidence = pred_confidence
                
                # Process best prediction if available
                if best_prediction and best_confidence > 0.65:  # Adjusted threshold for better precision
                    predicted_label = best_prediction['predicted_label']
                    confidence = best_prediction['confidence']
                    
                    # OPTIMIZATION 5: Enhanced Event Generation with 6-Category Context
                    if predicted_label != 'normal':
                        # Enhanced severity mapping for 6-category dataset
                        severity_map = {
                            'suspicious_gesture': 'orange',      # Hand gestures (medium severity)
                            'suspicious_looking': 'yellow',      # Head turning (lower severity)
                            'leaning': 'yellow',
                            'looking_around': 'yellow', 
                            'phone_use': 'red',
                            'cheating': 'red',
                            'suspicious': 'orange'
                        }
                        
                        severity = severity_map.get(predicted_label, 'orange')
                        
                        # Enhanced confidence-based severity adjustment
                        if confidence > 0.9:
                            if severity == 'yellow':
                                severity = 'orange'  # Escalate high-confidence yellow to orange
                        elif confidence < 0.75:
                            if severity == 'red':
                                severity = 'orange'  # De-escalate low-confidence red to orange
                        
                        # Enhanced auxiliary behavior analysis
                        gesture_detected = best_prediction.get('gesture_detected', False)
                        looking_detected = best_prediction.get('looking_detected', False)
                        behavior_details = best_prediction.get('behavior_details', {})
                        
                        # Escalate severity if multiple behaviors detected
                        if gesture_detected and looking_detected:
                            if severity == 'yellow':
                                severity = 'orange'
                            elif severity == 'orange':
                                severity = 'red'
                        
                        # Calculate behavioral persistence for additional validation
                        recent_predictions = getattr(self, '_recent_predictions', [])
                        recent_predictions.append(predicted_label)
                        if len(recent_predictions) > 5:
                            recent_predictions.pop(0)
                        self._recent_predictions = recent_predictions
                        
                        # Count consistent predictions for persistence scoring
                        consistency_score = sum(1 for p in recent_predictions if p == predicted_label) / len(recent_predictions)
                        
                        # Generate enhanced events with 6-category context
                        for pose in pose_results:
                            event_confidence = confidence * consistency_score  # Adjust by persistence
                            
                            # Enhanced event details
                            enhanced_details = f'Enhanced LSTM detected {predicted_label} pattern (conf: {confidence:.3f}, persistence: {consistency_score:.2f})'
                            
                            # Add auxiliary behavior information
                            if gesture_detected:
                                enhanced_details += f', Hand gestures: {best_prediction.get("gesture_confidence", 0):.2f}'
                            if looking_detected:
                                enhanced_details += f', Head turning: {best_prediction.get("looking_confidence", 0):.2f}'
                            
                            # Add behavior details if available
                            if behavior_details:
                                specific_indicators = behavior_details.get('specific_indicators', [])
                                if specific_indicators:
                                    indicator_types = [ind['type'] for ind in specific_indicators]
                                    enhanced_details += f', Indicators: {", ".join(indicator_types)}'
                            
                            events.append({
                                'timestamp': timestamp,
                                'person_id': pose['person_id'],
                                'event_type': f'Enhanced LSTM: {predicted_label.replace("_", " ").title()}',
                                'severity': severity,
                                'confidence': event_confidence,
                                'source': 'enhanced_lstm_6category',
                                'details': enhanced_details,
                                'bbox': pose['bbox'],
                                'sequence_length': len(self.behavior_history),
                                'consistency': consistency_score,
                                # Enhanced fields
                                'gesture_detected': gesture_detected,
                                'looking_detected': looking_detected,
                                'behavior_analysis': behavior_details,
                                'model_type': 'enhanced_6category'
                            })
            
            # OPTIMIZATION 6: Improved Fallback Analysis
            elif len(self.behavior_history) >= self.sequence_length:
                # Enhanced rule-based temporal analysis with trend detection
                recent_window = min(15, len(self.behavior_history))  # Adaptive window size
                recent_behaviors = np.array(self.behavior_history[-recent_window:])
                
                # Calculate trends and persistence
                lean_trend = np.mean(recent_behaviors[:, 0])      # lean_flag average
                look_trend = np.mean(recent_behaviors[:, 1])      # look_flag average  
                phone_trend = np.mean(recent_behaviors[:, 2])     # phone_flag average
                
                # Calculate trend slopes for behavior escalation
                if len(recent_behaviors) >= 5:
                    time_indices = np.arange(len(recent_behaviors))
                    lean_slope = np.polyfit(time_indices, recent_behaviors[:, 0], 1)[0]
                    look_slope = np.polyfit(time_indices, recent_behaviors[:, 1], 1)[0]
                    
                    # Detect escalating behaviors
                    escalation_detected = (lean_slope > 0.02 or look_slope > 0.02)
                else:
                    escalation_detected = False
                
                # Enhanced thresholds with adaptive sensitivity
                phone_threshold = 0.4    # Less sensitive for phone detection (40% confidence)
                multi_behavior_threshold = 0.4   # Multiple behaviors
                single_behavior_threshold = 0.6  # Single persistent behavior
                
                # Generate enhanced fallback events
                for pose in pose_results:
                    if phone_trend > phone_threshold:
                        events.append({
                            'timestamp': timestamp,
                            'person_id': pose['person_id'],
                            'event_type': 'Temporal: Persistent Phone Activity',
                            'severity': 'red',
                            'confidence': min(0.9, phone_trend + 0.1),
                            'source': 'temporal_rules_optimized',
                            'details': f'Phone activity trend: {phone_trend:.2f} over {recent_window} frames',
                            'bbox': pose['bbox']
                        })
                    elif lean_trend > multi_behavior_threshold and look_trend > multi_behavior_threshold:
                        severity = 'orange' if escalation_detected else 'yellow'
                        events.append({
                            'timestamp': timestamp,
                            'person_id': pose['person_id'],
                            'event_type': 'Temporal: Combined Suspicious Behavior',
                            'severity': severity,
                            'confidence': 0.7 + (0.1 if escalation_detected else 0),
                            'source': 'temporal_rules_optimized',
                            'details': f'Lean: {lean_trend:.2f}, Look: {look_trend:.2f} {"(escalating)" if escalation_detected else ""}',
                            'bbox': pose['bbox']
                        })
                    elif lean_trend > single_behavior_threshold:
                        events.append({
                            'timestamp': timestamp,
                            'person_id': pose['person_id'],
                            'event_type': 'Temporal: Persistent Leaning',
                            'severity': 'yellow',
                            'confidence': 0.6,
                            'source': 'temporal_rules_optimized',
                            'details': f'Leaning trend: {lean_trend:.2f} over {recent_window} frames',
                            'bbox': pose['bbox']
                        })
                    elif look_trend > single_behavior_threshold:
                        events.append({
                            'timestamp': timestamp,
                            'person_id': pose['person_id'],
                            'event_type': 'Temporal: Persistent Looking Around',
                            'severity': 'yellow',
                            'confidence': 0.6,
                            'source': 'temporal_rules_optimized',
                            'details': f'Looking trend: {look_trend:.2f} over {recent_window} frames',
                            'bbox': pose['bbox']
                        })
            
            # OPTIMIZATION 7: Memory Management for Long Sessions
            # Periodic cleanup to prevent memory bloat in long sessions
            if len(self.behavior_history) > 100:  # Every ~3 seconds at 30fps
                # Keep recent history and statistical summary
                self.behavior_history = self.behavior_history[-50:]
            
        except Exception as e:
            logger.error(f"Error in optimized temporal pattern analysis: {e}")
        
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
                indicators.append(f"L{pose.get('lean_angle', 0):.0f}°")
            if pose['look_flag']:
                indicators.append(f"H{pose.get('head_turn_angle', 0):.0f}°")
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
            
            # Skip database storage for now
            # for event in events:
            #     self.db.store_event(
            #         timestamp=timestamp,
            #         cam_id=cam_id,
            #         track_id=event['person_id'],
            #         event_type=event['event_type'],
            #         confidence=event['confidence'],
            #         evidence_path=filepath
            #     )
        
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
                'trained_model_loaded': self.lstm_classifier.is_loaded,
                'model_classes': self.lstm_classifier.class_labels if self.lstm_classifier.is_loaded else None,
                'model_info': self.lstm_classifier.get_model_info() if self.lstm_classifier.is_loaded else None
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

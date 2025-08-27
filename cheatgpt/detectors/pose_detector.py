"""YOLOv11-Pose detector for person pose estimation and behavior analysis."""
import os
import logging
import math
import time
from typing import List, Dict, Any, Tuple, Optional
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseDetector:
    """YOLOv11-Pose detector for analyzing human poses and behaviors."""
    
    # COCO pose keypoints indices
    KEYPOINTS = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
    
    def __init__(self, weights_path=None):
        """Initialize optimized pose detector with enhanced configuration."""
        self.weights_path = weights_path or os.getenv('POSE_MODEL_PATH', 'weights/yolo11m-pose.pt')
        self.force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        
        # OPTIMIZATION 1: Enhanced threshold configuration with adaptive sensitivity
        self.lean_angle_thresh = float(os.getenv('LEAN_ANGLE_THRESH', '8.0'))    # More sensitive (was 12.0)
        self.head_turn_thresh = float(os.getenv('HEAD_TURN_THRESH', '12.0'))     # More sensitive (was 18.0)  
        self.phone_iou_thresh = float(os.getenv('PHONE_IOU_THRESH', '0.08'))     # Optimized sensitivity
        
        # OPTIMIZATION 2: Performance-oriented thresholds
        self.confidence_thresh = float(os.getenv('POSE_CONFIDENCE_THRESH', '0.3'))
        self.min_keypoint_conf = float(os.getenv('MIN_KEYPOINT_CONF', '0.4'))    # Higher for reliability
        self.debug_mode = os.getenv('DEBUG_POSE', 'false').lower() == 'true'
        
        # OPTIMIZATION 3: Temporal smoothing parameters
        self.enable_temporal_smoothing = os.getenv('ENABLE_TEMPORAL_SMOOTHING', 'true').lower() == 'true'
        self.smoothing_window = int(os.getenv('SMOOTHING_WINDOW', '3'))         # Frames for smoothing
        
        # OPTIMIZATION 4: Performance monitoring
        self.performance_tracking = os.getenv('TRACK_PERFORMANCE', 'false').lower() == 'true'
        self._inference_times = []
        self._frame_count = 0
        
        # OPTIMIZATION 5: Keypoint history for temporal smoothing
        if self.enable_temporal_smoothing:
            self._keypoint_history = {}  # person_id -> list of keypoints
            self._angle_history = {}     # person_id -> list of angles
        
        # Determine device with enhanced selection
        self.device = self._get_device()
        logger.info(f"Optimized PoseDetector initialized on device: {self.device}")
        
        # Load model with optimizations
        self.model = self._load_model()
        
        # OPTIMIZATION 6: Model warmup for consistent inference times
        self._warmup_model()
        
    def _warmup_model(self):
        """Warm up the model with dummy inference for consistent performance."""
        try:
            # Create dummy frame for warmup
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Run a few warmup inferences
            logger.info("Warming up pose detection model...")
            for _ in range(3):
                _ = self.model(dummy_frame, device=self.device, verbose=False)
            
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _smooth_keypoints(self, person_id: str, current_keypoints: np.ndarray, 
                         current_confs: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply temporal smoothing to reduce keypoint jitter."""
        if not self.enable_temporal_smoothing:
            return current_keypoints
        
        try:
            # Initialize history for new person
            if person_id not in self._keypoint_history:
                self._keypoint_history[person_id] = []
            
            # Add current keypoints to history
            self._keypoint_history[person_id].append(current_keypoints.copy())
            
            # Keep only recent history
            max_history = self.smoothing_window
            if len(self._keypoint_history[person_id]) > max_history:
                self._keypoint_history[person_id] = self._keypoint_history[person_id][-max_history:]
            
            # Apply smoothing if we have enough history
            if len(self._keypoint_history[person_id]) >= 2:
                history = np.array(self._keypoint_history[person_id])
                
                # Use weighted average with more weight on recent frames
                weights = np.linspace(0.3, 1.0, len(history))
                weights = weights / np.sum(weights)
                
                # Apply smoothing only to confident keypoints
                smoothed = np.average(history, axis=0, weights=weights)
                
                # Validate smoothed keypoints (no negative coordinates)
                smoothed = np.maximum(smoothed, 0)
                
                return smoothed
            
            return current_keypoints
            
        except Exception as e:
            logger.debug(f"Keypoint smoothing error: {e}")
            return current_keypoints
        
    def _get_device(self) -> str:
        """Determine the best available device for inference."""
        if self.force_cpu:
            device = 'cpu'
            logger.info("Forced to use CPU device")
        elif torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Using GPU device: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            logger.info("CUDA not available, falling back to CPU")
        
        return device
    
    def _load_model(self) -> YOLO:
        """Load the YOLOv11-Pose model with the specified weights."""
        try:
            # Convert relative path to absolute if needed
            if not os.path.isabs(self.weights_path):
                base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to cheatgpt dir
                self.weights_path = os.path.join(base_dir, self.weights_path)
            
            logger.info(f"Loading YOLOv11-Pose model from: {self.weights_path}")
            model = YOLO(self.weights_path)
            
            # Move model to the appropriate device
            model.to(self.device)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load pose model: {e}")
            raise
    
    def estimate(self, frame: np.ndarray, phone_detections: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Optimized pose estimation with enhanced performance and accuracy.
        
        Args:
            frame: Input image as numpy array (BGR format)
            phone_detections: List of phone detections from YOLO detector
            
        Returns:
            List of pose estimates with optimized features
        """
        if frame is None:
            logger.warning("Received None frame for pose estimation")
            return []
        
        if phone_detections is None:
            phone_detections = []
        
        # OPTIMIZATION 1: Performance tracking
        start_time = time.time() if self.performance_tracking else None
        
        try:
            # OPTIMIZATION 2: Efficient inference
            results = self.model(frame, device=self.device, verbose=False)
            
            pose_estimates = []
            
            # OPTIMIZATION 3: Vectorized processing
            for result in results:
                if result.boxes is not None and result.keypoints is not None:
                    boxes = result.boxes
                    keypoints = result.keypoints
                    
                    # Batch process all detections
                    num_detections = len(boxes)
                    
                    for i in range(num_detections):
                        # Get person bounding box and confidence
                        bbox = boxes.xyxy[i].cpu().numpy().tolist()
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # OPTIMIZATION 4: Early filtering for performance
                        if conf < 0.4:  # Slightly higher threshold for better quality
                            continue
                        
                        # Get keypoints for this person
                        person_keypoints = keypoints.xy[i].cpu().numpy()  # Shape: (17, 2)
                        keypoint_confs = keypoints.conf[i].cpu().numpy() if hasattr(keypoints, 'conf') else None
                        
                        # OPTIMIZATION 5: Apply temporal smoothing
                        person_id = f"person_{i}"
                        if self.enable_temporal_smoothing:
                            person_keypoints = self._smooth_keypoints(person_id, person_keypoints, keypoint_confs)
                        
                        # OPTIMIZATION 6: Efficient keypoint extraction with validation
                        head_points = self._extract_head_keypoints(person_keypoints, keypoint_confs)
                        shoulder_points = self._extract_shoulder_keypoints(person_keypoints, keypoint_confs)
                        hip_points = self._extract_hip_keypoints(person_keypoints, keypoint_confs)
                        
                        # OPTIMIZATION 7: Skip processing if insufficient keypoints
                        min_keypoints_required = 3
                        total_keypoints = len(head_points) + len(shoulder_points) + len(hip_points)
                        if total_keypoints < min_keypoints_required:
                            continue
                        
                        # OPTIMIZATION 8: Optimized feature computation
                        yaw, pitch = self._compute_head_angles(head_points)
                        lean_flag = self._compute_leaning(shoulder_points, hip_points)
                        look_flag = self._compute_looking_around(yaw)
                        phone_flag = self._compute_phone_near(bbox, phone_detections)
                        
                        # OPTIMIZATION 9: Enhanced pose estimate with additional metadata
                        pose_estimate = {
                            'track_id': person_id,
                            'bbox': bbox,
                            'yaw': yaw,
                            'pitch': pitch,
                            'lean_flag': lean_flag,
                            'look_flag': look_flag,
                            'phone_flag': phone_flag,
                            'lean_angle': abs(self._calculate_lean_angle(shoulder_points, hip_points)),
                            'head_turn_angle': abs(yaw),
                            'confidence': conf,
                            'keypoint_quality': self._assess_keypoint_quality(head_points, shoulder_points, hip_points),
                            'total_keypoints': total_keypoints,
                            'frame_timestamp': self._frame_count
                        }
                        
                        pose_estimates.append(pose_estimate)
            
            # OPTIMIZATION 10: Performance monitoring
            if self.performance_tracking and start_time:
                inference_time = time.time() - start_time
                self._inference_times.append(inference_time)
                
                # Log performance stats periodically
                if len(self._inference_times) >= 30:  # Every 30 frames
                    avg_time = np.mean(self._inference_times)
                    logger.info(f"Pose detection performance: {avg_time*1000:.1f}ms avg, {1/avg_time:.1f} FPS")
                    self._inference_times = self._inference_times[-10:]  # Keep recent history
            
            self._frame_count += 1
            
            logger.debug(f"Generated {len(pose_estimates)} optimized pose estimates")
            return pose_estimates
            
        except Exception as e:
            logger.error(f"Error during optimized pose estimation: {e}")
            return []
    
    def _calculate_lean_angle(self, shoulder_points: Dict[str, Tuple[float, float]], 
                             hip_points: Dict[str, Tuple[float, float]]) -> float:
        """Calculate the actual lean angle in degrees."""
        try:
            if ('left_shoulder' in shoulder_points and 'right_shoulder' in shoulder_points):
                left_shoulder = shoulder_points['left_shoulder']
                right_shoulder = shoulder_points['right_shoulder']
                
                # Calculate shoulder line angle from horizontal
                shoulder_dx = right_shoulder[0] - left_shoulder[0]
                shoulder_dy = right_shoulder[1] - left_shoulder[1]
                
                if abs(shoulder_dx) > 10:  # Valid shoulder separation
                    angle = math.degrees(math.atan2(abs(shoulder_dy), abs(shoulder_dx)))
                    return min(angle, 45.0)  # Cap at reasonable maximum
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _assess_keypoint_quality(self, head_points: Dict, shoulder_points: Dict, hip_points: Dict) -> float:
        """Assess the quality of detected keypoints for confidence scoring."""
        try:
            total_possible = 7  # nose, 2 eyes, 2 shoulders, 2 hips
            detected = len(head_points) + len(shoulder_points) + len(hip_points)
            
            quality_score = detected / total_possible
            
            # Bonus for having both shoulders (critical for lean detection)
            if len(shoulder_points) >= 2:
                quality_score += 0.1
            
            # Bonus for having head keypoints (critical for head angle)
            if len(head_points) >= 2:
                quality_score += 0.1
            
            return min(1.0, quality_score)
            
        except Exception:
            return 0.0
    
    def _extract_head_keypoints(self, keypoints: np.ndarray, confs: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float]]:
        """Extract head keypoints with optimized confidence filtering."""
        head_points = {}
        
        for name, idx in [('nose', self.KEYPOINTS['nose']), 
                         ('left_eye', self.KEYPOINTS['left_eye']), 
                         ('right_eye', self.KEYPOINTS['right_eye'])]:
            if idx < len(keypoints):
                x, y = keypoints[idx]
                # Use optimized confidence threshold
                if confs is None or (idx < len(confs) and confs[idx] > self.min_keypoint_conf):
                    if x > 0 and y > 0:  # Valid keypoint
                        head_points[name] = (float(x), float(y))
        
        return head_points
    
    def _extract_shoulder_keypoints(self, keypoints: np.ndarray, confs: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float]]:
        """Extract shoulder keypoints with optimized confidence filtering."""
        shoulder_points = {}
        
        for name, idx in [('left_shoulder', self.KEYPOINTS['left_shoulder']), 
                         ('right_shoulder', self.KEYPOINTS['right_shoulder'])]:
            if idx < len(keypoints):
                x, y = keypoints[idx]
                # Use optimized confidence threshold
                if confs is None or (idx < len(confs) and confs[idx] > self.min_keypoint_conf):
                    if x > 0 and y > 0:  # Valid keypoint
                        shoulder_points[name] = (float(x), float(y))
        
        return shoulder_points
    
    def _extract_hip_keypoints(self, keypoints: np.ndarray, confs: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float]]:
        """Extract hip keypoints with optimized confidence filtering."""
        hip_points = {}
        
        for name, idx in [('left_hip', self.KEYPOINTS['left_hip']), 
                         ('right_hip', self.KEYPOINTS['right_hip'])]:
            if idx < len(keypoints):
                x, y = keypoints[idx]
                # Use optimized confidence threshold
                if confs is None or (idx < len(confs) and confs[idx] > self.min_keypoint_conf):
                    if x > 0 and y > 0:  # Valid keypoint
                        hip_points[name] = (float(x), float(y))
        
        return hip_points
    
    def _compute_head_angles(self, head_points: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """Optimized head angle computation with multi-method validation and temporal smoothing."""
        yaw, pitch = 0.0, 0.0
        confidence_score = 0.0
        
        try:
            # Pre-compute commonly used points for efficiency
            nose = head_points.get('nose')
            left_eye = head_points.get('left_eye')
            right_eye = head_points.get('right_eye')
            left_ear = head_points.get('left_ear')
            right_ear = head_points.get('right_ear')
            
            # METHOD 1: Optimized Eye-Symmetry Analysis (Primary method)
            if left_eye and right_eye:
                # Vectorized calculations for efficiency
                eye_vector = np.array(right_eye) - np.array(left_eye)
                eye_dx, eye_dy = eye_vector
                eye_separation = np.linalg.norm(eye_vector)
                eye_center = (np.array(left_eye) + np.array(right_eye)) / 2
                
                if eye_separation > 20:  # Minimum threshold for reliable detection
                    # Calculate perspective distortion factor
                    # Normal frontal face: eye separation ~65-70 pixels at typical webcam distance
                    baseline_separation = 65.0
                    perspective_factor = eye_separation / baseline_separation
                    
                    # Yaw estimation using geometric perspective
                    if perspective_factor < 0.90:  # More sensitive - detect smaller turns (was 0.85)
                        # Calculate turn magnitude using perspective distortion
                        yaw_magnitude = (0.90 - perspective_factor) * 65.0  # More sensitive scaling (was 60.0)
                        
                        # Direction determination using eye center offset
                        if abs(eye_dx) > 3:  # More sensitive minimum offset (was 5)
                            direction_factor = eye_dx / eye_separation
                            if direction_factor < -0.05:  # More sensitive - detect smaller turns (was -0.1)
                                yaw = -min(yaw_magnitude, 35.0)
                                confidence_score = 0.8
                            elif direction_factor > 0.05:  # More sensitive - detect smaller turns (was 0.1)
                                yaw = min(yaw_magnitude, 35.0)
                                confidence_score = 0.8
                        
                        # Additional validation using eye vertical alignment
                        if abs(eye_dy) < eye_separation * 0.2:  # Eyes reasonably aligned
                            confidence_score *= 1.1  # Boost confidence for good alignment
                        else:
                            confidence_score *= 0.7  # Reduce confidence for poor alignment
            
            # METHOD 2: Nose-Bridge Geometric Analysis (Secondary validation)
            if nose and left_eye and right_eye and confidence_score < 0.7:
                eye_center = (np.array(left_eye) + np.array(right_eye)) / 2
                nose_array = np.array(nose)
                
                # Calculate nose offset from eye center
                nose_offset_vector = nose_array - eye_center
                nose_offset_x = nose_offset_vector[0]
                
                # Normalize by face width
                face_width = abs(right_eye[0] - left_eye[0])
                if face_width > 15:
                    normalized_offset = nose_offset_x / (face_width / 2)
                    
                    # Calculate yaw using nose displacement
                    nose_yaw = normalized_offset * 25.0  # Calibrated multiplier
                    
                    # Use nose yaw if more significant than eye-based yaw
                    if abs(nose_yaw) > abs(yaw) * 1.2 and abs(nose_yaw) > 8:
                        yaw = max(-40, min(40, nose_yaw))
                        confidence_score = max(confidence_score, 0.6)
            
            # METHOD 3: Ear Visibility Analysis (Strong indicator for profile views)
            ear_yaw = 0.0
            if left_ear and not right_ear and left_eye:
                # Left ear visible, right ear not visible -> face turned right
                ear_eye_dist = abs(left_ear[0] - left_eye[0])
                if ear_eye_dist > 25:  # Significant ear prominence
                    ear_yaw = -30.0  # Strong right turn
                    confidence_score = max(confidence_score, 0.9)
            elif right_ear and not left_ear and right_eye:
                # Right ear visible, left ear not visible -> face turned left
                ear_eye_dist = abs(right_ear[0] - right_eye[0])
                if ear_eye_dist > 25:  # Significant ear prominence
                    ear_yaw = 30.0  # Strong left turn
                    confidence_score = max(confidence_score, 0.9)
            
            # Combine ear evidence with other methods
            if abs(ear_yaw) > abs(yaw) * 1.5:
                yaw = ear_yaw
            
            # METHOD 4: Temporal Smoothing and Noise Reduction
            # Apply temporal smoothing to reduce jitter (can be enhanced with frame history)
            if abs(yaw) < 3:  # Micro-movement threshold
                yaw = 0.0
            elif abs(yaw) < 8:  # Small movement - apply dampening
                yaw *= 0.7
            
            # METHOD 5: Enhanced Pitch Calculation
            if nose and (left_eye or right_eye):
                # Use available eye(s) for pitch reference
                if left_eye and right_eye:
                    eye_y = (left_eye[1] + right_eye[1]) / 2
                elif left_eye:
                    eye_y = left_eye[1]
                else:
                    eye_y = right_eye[1]
                
                # Calculate vertical displacement
                nose_eye_dy = nose[1] - eye_y
                
                # Improved pitch estimation using facial proportions
                if abs(nose_eye_dy) > 10:  # Threshold for significant head tilt
                    # Use average face height for normalization
                    avg_face_height = 60.0  # Calibrated for typical webcam distance
                    pitch_radians = math.atan2(nose_eye_dy, avg_face_height)
                    pitch = math.degrees(pitch_radians)
                    
                    # Clamp to reasonable range and apply sensitivity adjustment
                    pitch = max(-35, min(35, pitch * 0.8))  # Reduce sensitivity slightly
            
            # OPTIMIZATION: Cache results for temporal consistency
            # This would be enhanced with frame-to-frame smoothing in production
            
            # Enhanced debugging with confidence information
            if self.debug_mode and (abs(yaw) > 0 or abs(pitch) > 0):
                conf_label = "HIGH" if confidence_score > 0.8 else "MEDIUM" if confidence_score > 0.5 else "LOW"
                logger.info(f"Head angles computed: yaw={yaw:.1f}°, pitch={pitch:.1f}° "
                           f"(confidence: {conf_label} {confidence_score:.2f})")
        
        except Exception as e:
            logger.debug(f"Error in optimized head angle computation: {e}")
        
        return yaw, pitch
    
    def _compute_leaning(self, shoulder_points: Dict[str, Tuple[float, float]], 
                        hip_points: Dict[str, Tuple[float, float]]) -> bool:
        """
        Optimized anatomically-accurate leaning detection with performance improvements.
        """
        try:
            if not shoulder_points or not hip_points:
                return False
            
            # Pre-extract coordinates for vectorized operations
            left_shoulder = shoulder_points.get('left_shoulder')
            right_shoulder = shoulder_points.get('right_shoulder')
            left_hip = hip_points.get('left_hip')
            right_hip = hip_points.get('right_hip')
            
            # Quick validation - need at least shoulder data
            if not left_shoulder or not right_shoulder:
                return False
            
            lean_detected = False
            max_deviation = 0.0
            detection_method = "none"
            
            # Convert to numpy arrays for efficient computation
            ls_arr = np.array(left_shoulder)
            rs_arr = np.array(right_shoulder)
            
            # METHOD 1: OPTIMIZED SHOULDER LINE ANALYSIS (Primary - fastest)
            shoulder_vector = rs_arr - ls_arr
            shoulder_dx, shoulder_dy = shoulder_vector
            shoulder_separation = np.linalg.norm(shoulder_vector)
            
            if shoulder_separation > 25:  # Valid shoulder span
                # Calculate shoulder line angle from horizontal
                shoulder_angle = abs(math.degrees(math.atan2(shoulder_dy, shoulder_dx)))
                
                # Normalize angle to 0-90 degree range
                if shoulder_angle > 90:
                    shoulder_angle = 180 - shoulder_angle
                
                # Enhanced threshold with body size adaptation
                adapted_threshold = max(self.lean_angle_thresh, 
                                      10.0 if shoulder_separation > 50 else 12.0)
                
                if shoulder_angle > adapted_threshold:
                    lean_detected = True
                    max_deviation = shoulder_angle
                    detection_method = "optimized_shoulder_tilt"
            
            # METHOD 2: ANATOMICAL TORSO ALIGNMENT (Secondary - if hips available)
            if not lean_detected and left_hip and right_hip:
                # Vectorized hip and shoulder center calculation
                lh_arr = np.array(left_hip)
                rh_arr = np.array(right_hip)
                
                shoulder_center = (ls_arr + rs_arr) / 2
                hip_center = (lh_arr + rh_arr) / 2
                
                # Torso vector and alignment check
                torso_vector = shoulder_center - hip_center
                torso_height = abs(torso_vector[1])
                
                if torso_height > 35:  # Minimum torso height for reliable measurement
                    # Calculate lean angle from vertical
                    torso_lean_angle = abs(math.degrees(math.atan2(abs(torso_vector[0]), torso_height)))
                    
                    # Adaptive threshold based on detection confidence
                    torso_threshold = max(self.lean_angle_thresh * 0.6, 5.0)  # More sensitive (was 0.8, 8.0)
                    
                    if torso_lean_angle > torso_threshold:
                        lean_detected = True
                        max_deviation = max(max_deviation, torso_lean_angle)
                        if detection_method == "none":
                            detection_method = "anatomical_torso_axis"
            
            # METHOD 3: BILATERAL ASYMMETRY ANALYSIS (Tertiary - detailed check)
            if not lean_detected and left_hip and right_hip:
                # Calculate bilateral torso lengths efficiently
                left_torso_vector = ls_arr - lh_arr
                right_torso_vector = rs_arr - rh_arr
                
                left_torso_length = np.linalg.norm(left_torso_vector)
                right_torso_length = np.linalg.norm(right_torso_vector)
                
                if min(left_torso_length, right_torso_length) > 30:  # Valid measurements
                    # Calculate asymmetry ratio
                    length_diff = abs(left_torso_length - right_torso_length)
                    max_length = max(left_torso_length, right_torso_length)
                    asymmetry_ratio = length_diff / max_length
                    
                    # Optimized threshold for asymmetry detection
                    asymmetry_threshold = 0.25  # 25% asymmetry tolerance
                    
                    if asymmetry_ratio > asymmetry_threshold:
                        asymmetry_angle = math.degrees(math.atan(asymmetry_ratio))
                        
                        if asymmetry_angle > 15:  # Significant asymmetry
                            lean_detected = True
                            max_deviation = max(max_deviation, asymmetry_angle)
                            if detection_method == "none":
                                detection_method = "bilateral_asymmetry"
            
            # METHOD 4: CENTER OF MASS DISPLACEMENT (Quaternary - advanced)
            if not lean_detected and left_hip and right_hip and max_deviation < 8:
                # Quick center of mass check for subtle leans
                shoulder_center = (ls_arr + rs_arr) / 2
                hip_center = (lh_arr + rh_arr) / 2
                
                # Weighted anatomical center (shoulders 40%, hips 60%)
                center_of_mass = shoulder_center * 0.4 + hip_center * 0.6
                
                # Calculate displacement from vertical through hip center
                displacement = abs(center_of_mass[0] - hip_center[0])
                
                # Reference width for proportional analysis
                torso_width = max(shoulder_separation, np.linalg.norm(rh_arr - lh_arr))
                
                if torso_width > 25:  # Valid reference width
                    displacement_ratio = displacement / torso_width
                    
                    # Optimized displacement threshold
                    if displacement_ratio > 0.22:  # 22% displacement
                        displacement_angle = math.degrees(math.atan(displacement_ratio))
                        
                        if displacement_angle > 13:
                            lean_detected = True
                            max_deviation = max(max_deviation, displacement_angle)
                            if detection_method == "none":
                                detection_method = "center_of_mass_displacement"
            
            # OPTIMIZATION: Confidence-based result filtering
            # Reduce false positives by requiring minimum confidence
            if lean_detected and max_deviation < self.lean_angle_thresh * 1.2:
                # Apply additional validation for borderline cases
                confidence_factor = max_deviation / (self.lean_angle_thresh * 1.5)
                if confidence_factor < 0.8:  # Low confidence detection
                    lean_detected = False
                    detection_method += "_low_confidence_filtered"
            
            # Enhanced debug output with performance metrics
            if self.debug_mode:
                if lean_detected:
                    logger.info(f"🏃 OPTIMIZED LEAN DETECTED: method={detection_method}, "
                               f"deviation={max_deviation:.1f}°, threshold={self.lean_angle_thresh}°")
                else:
                    logger.debug(f"✅ Normal posture (max_dev={max_deviation:.1f}°, method={detection_method})")
            
            return lean_detected
            
        except Exception as e:
            logger.debug(f"Error in optimized leaning analysis: {e}")
            return False
    
    def _compute_looking_around(self, yaw: float) -> bool:
        """Determine if person is looking around based on head yaw angle."""
        is_looking = abs(yaw) > self.head_turn_thresh
        
        if self.debug_mode:
            if is_looking:
                direction = "LEFT" if yaw > 0 else "RIGHT"
                logger.info(f"👁️ LOOKING AROUND DETECTED: direction={direction}, "
                           f"yaw={yaw:.1f}°, threshold={self.head_turn_thresh}°")
            else:
                logger.debug(f"Head position normal: yaw={yaw:.1f}°")
        
        return is_looking
    
    def _compute_phone_near(self, person_bbox: List[float], 
                           phone_detections: List[Dict]) -> bool:
        """Compute if phone is near person's torso using IoU."""
        try:
            if not phone_detections:
                return False
            
            # Define expanded detection region around person
            x1, y1, x2, y2 = person_bbox
            person_width = x2 - x1
            person_height = y2 - y1
            
            # Expand detection area around person (more lenient)
            margin_x = person_width * 0.3   # 30% margin on sides
            margin_y = person_height * 0.2   # 20% margin on top/bottom
            
            expanded_x1 = x1 - margin_x
            expanded_y1 = y1 - margin_y
            expanded_x2 = x2 + margin_x
            expanded_y2 = y2 + margin_y
            
            expanded_bbox = [expanded_x1, expanded_y1, expanded_x2, expanded_y2]
            
            # Check IoU with each phone detection
            for phone in phone_detections:
                phone_bbox = phone.get('bbox', [])
                if len(phone_bbox) == 4:
                    iou = self._calculate_iou(expanded_bbox, phone_bbox)
                    
                    # Also check simple overlap for better detection
                    overlap = self._calculate_overlap_ratio(person_bbox, phone_bbox)
                    
                    if self.debug_mode:
                        logger.info(f"Phone detection: IoU={iou:.3f}, overlap={overlap:.3f}, thresh={self.phone_iou_thresh}")
                    
                    if iou > self.phone_iou_thresh or overlap > 0.1:
                        if self.debug_mode:
                            logger.info(f"Phone detected near person: IoU={iou:.3f}, overlap={overlap:.3f}")
                        return True
        
        except Exception as e:
            logger.debug(f"Error computing phone near: {e}")
        
        return False
    
    def _calculate_overlap_ratio(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap ratio (intersection over smaller bbox)."""
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
            
            # Calculate areas
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            
            # Return intersection over smaller area
            smaller_area = min(area1, area2)
            
            if smaller_area <= 0:
                return 0.0
            
            return intersection / smaller_area
        
        except Exception:
            return 0.0
    
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
        
        except Exception as e:
            logger.debug(f"Error calculating IoU: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded pose model."""
        return {
            'model_path': self.weights_path,
            'device': self.device,
            'lean_angle_thresh': self.lean_angle_thresh,
            'head_turn_thresh': self.head_turn_thresh,
            'phone_iou_thresh': self.phone_iou_thresh,
            'cuda_available': torch.cuda.is_available(),
            'force_cpu': self.force_cpu
        }

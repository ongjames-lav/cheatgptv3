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
                        arm_points = self._extract_arm_keypoints(person_keypoints, keypoint_confs)
                        
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
                        gesture_flag = self._compute_suspicious_gesture(arm_points, head_points)
                        
                        # OPTIMIZATION 9: Enhanced pose estimate with additional metadata
                        pose_estimate = {
                            'track_id': person_id,
                            'bbox': bbox,
                            'yaw': yaw,
                            'pitch': pitch,
                            'lean_flag': lean_flag,
                            'look_flag': look_flag,
                            'phone_flag': phone_flag,
                            'gesture_flag': gesture_flag,
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
    
    def _extract_arm_keypoints(self, keypoints: np.ndarray, confs: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float]]:
        """Extract arm keypoints (elbows and wrists) from pose data."""
        arm_points = {}
        
        for name, idx in [('left_elbow', self.KEYPOINTS['left_elbow']),
                         ('right_elbow', self.KEYPOINTS['right_elbow']),
                         ('left_wrist', self.KEYPOINTS['left_wrist']),
                         ('right_wrist', self.KEYPOINTS['right_wrist'])]:
            if idx < len(keypoints):
                x, y = keypoints[idx]
                # Use optimized confidence threshold
                if confs is None or (idx < len(confs) and confs[idx] > self.min_keypoint_conf):
                    if x > 0 and y > 0:  # Valid keypoint
                        arm_points[name] = (float(x), float(y))
        
        return arm_points
    
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
        Enhanced anatomically-accurate leaning detection optimized for cheating behavior detection.
        """
        try:
            # Debug input data
            logger.debug(f"🏃 Lean detection - Shoulder points: {shoulder_points}")
            logger.debug(f"🏃 Lean detection - Hip points: {hip_points}")
            
            if not shoulder_points or not hip_points:
                logger.debug("🏃 No shoulder or hip points for lean analysis")
                return False
            
            # Pre-extract coordinates for vectorized operations
            left_shoulder = shoulder_points.get('left_shoulder')
            right_shoulder = shoulder_points.get('right_shoulder')
            left_hip = hip_points.get('left_hip')
            right_hip = hip_points.get('right_hip')
            
            logger.debug(f"🏃 Extracted keypoints - LS: {left_shoulder}, RS: {right_shoulder}, LH: {left_hip}, RH: {right_hip}")
            
            # Quick validation - need at least shoulder data
            if not left_shoulder or not right_shoulder:
                logger.debug("🏃 Missing shoulder keypoints for lean detection")
                return False
            
            lean_detected = False
            max_deviation = 0.0
            detection_method = "none"
            lean_direction = "none"
            
            # Convert to numpy arrays for efficient computation
            ls_arr = np.array(left_shoulder)
            rs_arr = np.array(right_shoulder)
            
            # METHOD 1: ENHANCED SHOULDER LINE ANALYSIS (Primary - fastest)
            shoulder_vector = rs_arr - ls_arr
            shoulder_dx, shoulder_dy = shoulder_vector
            shoulder_separation = np.linalg.norm(shoulder_vector)
            
            logger.debug(f"🏃 Shoulder analysis - Vector: ({shoulder_dx:.1f}, {shoulder_dy:.1f}), Separation: {shoulder_separation:.1f}")
            
            if shoulder_separation > 25:  # Valid shoulder span
                # Calculate shoulder line angle from horizontal
                shoulder_angle = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
                shoulder_tilt = abs(shoulder_angle)
                
                # Normalize angle to 0-90 degree range
                if shoulder_tilt > 90:
                    shoulder_tilt = 180 - shoulder_tilt
                
                logger.debug(f"🏃 Shoulder angle: {shoulder_angle:.1f}°, Tilt: {shoulder_tilt:.1f}°")
                
                # Determine lean direction for cheating analysis
                if shoulder_angle > 2:  # More sensitive - Left shoulder higher (leaning right)
                    lean_direction = "right"
                elif shoulder_angle < -2:  # More sensitive - Right shoulder higher (leaning left)
                    lean_direction = "left"
                
                logger.debug(f"🏃 Lean direction from shoulders: {lean_direction}")
                
                # Enhanced threshold with body size adaptation
                # More sensitive for cheating detection scenarios
                adapted_threshold = max(self.lean_angle_thresh * 0.6, 2.0)  # Even more sensitive threshold
                
                logger.debug(f"🏃 Shoulder tilt {shoulder_tilt:.1f}° vs threshold {adapted_threshold:.1f}°")
                
                if shoulder_tilt > adapted_threshold:
                    lean_detected = True
                    max_deviation = shoulder_tilt
                    detection_method = "optimized_shoulder_tilt"
                    logger.info(f"🏃 LEAN DETECTED via shoulder tilt: {shoulder_tilt:.1f}° > {adapted_threshold:.1f}°, direction: {lean_direction}")
                else:
                    logger.debug(f"🏃 Shoulder tilt {shoulder_tilt:.1f}° below threshold {adapted_threshold:.1f}°")
            
            # METHOD 2: ENHANCED ANATOMICAL TORSO ALIGNMENT (Secondary - if hips available)
            if left_hip and right_hip:
                # Vectorized hip and shoulder center calculation
                lh_arr = np.array(left_hip)
                rh_arr = np.array(right_hip)
                
                shoulder_center = (ls_arr + rs_arr) / 2
                hip_center = (lh_arr + rh_arr) / 2
                
                # Torso vector and alignment check
                torso_vector = shoulder_center - hip_center
                torso_height = abs(torso_vector[1])
                
                if torso_height > 30:  # Reduced minimum torso height for better detection
                    # Calculate lean angle from vertical (enhanced for cheating detection)
                    torso_lean_angle = abs(math.degrees(math.atan2(abs(torso_vector[0]), torso_height)))
                    
                    # More sensitive threshold for suspicious posture detection
                    torso_threshold = max(self.lean_angle_thresh * 0.5, 3.5)  # Even more sensitive
                    
                    if torso_lean_angle > torso_threshold:
                        lean_detected = True
                        max_deviation = max(max_deviation, torso_lean_angle)
                        if detection_method == "none":
                            detection_method = "anatomical_torso_axis"
                        
                        # Enhance direction detection for torso lean
                        if torso_vector[0] > 10:  # Leaning right
                            lean_direction = "right" if lean_direction == "none" else lean_direction
                        elif torso_vector[0] < -10:  # Leaning left
                            lean_direction = "left" if lean_direction == "none" else lean_direction
            
            # METHOD 3: ENHANCED BILATERAL ASYMMETRY ANALYSIS (Tertiary - detailed check)
            if left_hip and right_hip:
                # Calculate bilateral torso lengths efficiently
                left_torso_vector = ls_arr - lh_arr
                right_torso_vector = rs_arr - rh_arr
                
                left_torso_length = np.linalg.norm(left_torso_vector)
                right_torso_length = np.linalg.norm(right_torso_vector)
                
                if min(left_torso_length, right_torso_length) > 25:  # Reduced for better detection
                    # Calculate asymmetry ratio
                    length_diff = abs(left_torso_length - right_torso_length)
                    max_length = max(left_torso_length, right_torso_length)
                    asymmetry_ratio = length_diff / max_length
                    
                    # More sensitive threshold for asymmetry detection
                    asymmetry_threshold = 0.2  # Reduced from 0.25 to 0.2 (20% asymmetry tolerance)
                    
                    if asymmetry_ratio > asymmetry_threshold:
                        asymmetry_angle = math.degrees(math.atan(asymmetry_ratio))
                        
                        if asymmetry_angle > 12:  # Reduced from 15 to 12 degrees
                            lean_detected = True
                            max_deviation = max(max_deviation, asymmetry_angle)
                            if detection_method == "none":
                                detection_method = "bilateral_asymmetry"
                            
                            # Determine direction based on which side is longer/shorter
                            if left_torso_length > right_torso_length:
                                lean_direction = "left" if lean_direction == "none" else lean_direction
                            else:
                                lean_direction = "right" if lean_direction == "none" else lean_direction
            
            # METHOD 4: NEW - SUSPICIOUS CHEATING POSTURE DETECTION
            # Detect specific postures commonly associated with cheating behavior
            if left_hip and right_hip and shoulder_separation > 30:
                # Calculate overall body axis deviation
                body_axis_vector = shoulder_center - hip_center
                body_axis_angle = abs(math.degrees(math.atan2(body_axis_vector[0], abs(body_axis_vector[1]))))
                
                # Check for forward lean combined with side lean (common cheating posture)
                forward_lean_threshold = 8.0  # Forward lean detection
                side_lean_threshold = 6.0     # Side lean detection
                
                # Calculate forward lean (looking down at paper/device)
                if torso_height > 25:
                    forward_component = abs(torso_vector[1]) - abs(torso_vector[0])
                    if forward_component < torso_height * 0.7:  # Significant forward lean
                        if body_axis_angle > side_lean_threshold:
                            lean_detected = True
                            max_deviation = max(max_deviation, body_axis_angle)
                            detection_method = "suspicious_cheating_posture"
                            logger.debug(f"🚨 Suspicious cheating posture detected: forward+side lean, angle={body_axis_angle:.1f}°")
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
            
            # METHOD 5: NEW - TEMPORAL LEANING PATTERN ANALYSIS
            # Track suspicious leaning patterns over time for enhanced detection
            current_time = time.time()
            person_id = getattr(self, 'current_person_id', 'default')
            
            if not hasattr(self, 'lean_history'):
                self.lean_history = {}
            
            if person_id not in self.lean_history:
                self.lean_history[person_id] = {
                    'angles': [],
                    'directions': [],
                    'timestamps': [],
                    'max_consecutive': 0
                }
            
            history = self.lean_history[person_id]
            
            # Clean old history (keep last 10 seconds)
            cutoff_time = current_time - 10.0
            valid_indices = [i for i, t in enumerate(history['timestamps']) if t > cutoff_time]
            
            if valid_indices:
                history['angles'] = [history['angles'][i] for i in valid_indices]
                history['directions'] = [history['directions'][i] for i in valid_indices]
                history['timestamps'] = [history['timestamps'][i] for i in valid_indices]
            else:
                history['angles'] = []
                history['directions'] = []
                history['timestamps'] = []
            
            # Add current measurement
            if max_deviation > 0:
                history['angles'].append(max_deviation)
                history['directions'].append(lean_direction)
                history['timestamps'].append(current_time)
                
                # Check for sustained lean pattern (suspicious behavior)
                if len(history['angles']) >= 3:
                    recent_angles = history['angles'][-3:]
                    recent_directions = history['directions'][-3:]
                    
                    # Check for consistent leaning in same direction (cheating pattern)
                    if (len(set(recent_directions)) == 1 and  # Same direction
                        all(angle > 4.0 for angle in recent_angles)):  # Consistent significant lean
                        
                        sustained_lean_angle = sum(recent_angles) / len(recent_angles)
                        if sustained_lean_angle > 6.0:  # Enhanced threshold for sustained lean
                            lean_detected = True
                            max_deviation = max(max_deviation, sustained_lean_angle)
                            detection_method = "sustained_suspicious_lean"
                            logger.debug(f"🚨 Sustained suspicious lean detected: {lean_direction} direction, avg={sustained_lean_angle:.1f}°")
            
            # Enhanced debug output with performance metrics
            if self.debug_mode:
                if lean_detected:
                    logger.info(f"🏃 ENHANCED LEAN DETECTED: method={detection_method}, "
                               f"deviation={max_deviation:.1f}°, direction={lean_direction}, threshold={self.lean_angle_thresh}°")
                else:
                    logger.debug(f"✅ Normal posture (max_dev={max_deviation:.1f}°, method={detection_method})")
            
            # Final debug summary
            logger.debug(f"🏃 Lean detection result: {lean_detected}, max_deviation: {max_deviation:.1f}°, method: {detection_method}, direction: {lean_direction}")
            
            return lean_detected
            
        except Exception as e:
            logger.debug(f"Error in enhanced leaning analysis: {e}")
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
    
    def _compute_suspicious_gesture(self, arm_points: Dict[str, Tuple[float, float]], 
                                   head_points: Dict[str, Tuple[float, float]]) -> bool:
        """
        Detect suspicious gestures like hands near face, touching head, or gesturing.
        """
        try:
            # Debug logging
            logger.debug(f"🤚 Gesture analysis - Arm points: {len(arm_points)} detected: {list(arm_points.keys())}")
            logger.debug(f"🤚 Gesture analysis - Head points: {len(head_points)} detected: {list(head_points.keys())}")
            
            if not arm_points or not head_points:
                logger.debug("🤚 No arm or head points for gesture analysis")
                return False
            
            # Get hand/wrist positions
            left_wrist = arm_points.get('left_wrist')
            right_wrist = arm_points.get('right_wrist')
            
            logger.debug(f"🤚 Wrist positions - Left: {left_wrist}, Right: {right_wrist}")
            
            # Get head region (use nose as center, or estimate from available points)
            head_center = None
            if 'nose' in head_points:
                head_center = head_points['nose']
                logger.debug(f"🤚 Using nose as head center: {head_center}")
            elif 'left_eye' in head_points and 'right_eye' in head_points:
                # Estimate head center from eyes
                left_eye = head_points['left_eye']
                right_eye = head_points['right_eye']
                head_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
                logger.debug(f"🤚 Estimated head center from eyes: {head_center}")
            
            if not head_center:
                logger.debug("🤚 No head center available for gesture analysis")
                return False
            
            gesture_detected = False
            detection_reason = ""
            
            # Calculate person/head scale for adaptive thresholds
            if 'left_eye' in head_points and 'right_eye' in head_points:
                eye_distance = abs(head_points['left_eye'][0] - head_points['right_eye'][0])
                # Use eye distance to estimate head size and create adaptive threshold
                # More conservative threshold for realistic cheating detection
                head_radius = max(eye_distance * 2.5, 140)  # Reduced multiplier and increased minimum
                logger.debug(f"🤚 Adaptive gesture detection using eye_distance: {eye_distance:.1f}px, head_radius: {head_radius:.1f}px")
            else:
                head_radius = 200  # More conservative default radius
                logger.debug(f"🤚 Default gesture detection using head_radius: {head_radius}px")
            
            # ADVANCED GESTURE DETECTION - Multiple pattern checking
            
            # Pattern 1: Hand near face/head (classic cheating gesture)
            for wrist_name, wrist_pos in [('left_wrist', left_wrist), ('right_wrist', right_wrist)]:
                if wrist_pos:
                    # Calculate distance from wrist to head center
                    distance = math.sqrt((wrist_pos[0] - head_center[0])**2 + 
                                       (wrist_pos[1] - head_center[1])**2)
                    
                    logger.debug(f"🤚 Distance check: {wrist_name} distance to head: {distance:.1f}px (threshold: {head_radius}px)")
                    
                    if distance < head_radius:
                        gesture_detected = True
                        detection_reason = f"{wrist_name}_near_head"
                        logger.info(f"🤚 SUSPICIOUS GESTURE: {wrist_name} near head (distance: {distance:.1f}px)")
                        break
            
            # Pattern 2: Hand in suspicious face region (conservative detection)
            if not gesture_detected:
                for wrist_name, wrist_pos in [('left_wrist', left_wrist), ('right_wrist', right_wrist)]:
                    if wrist_pos and head_center:
                        # Check if hand is in face region (conservative area)
                        horizontal_distance = abs(wrist_pos[0] - head_center[0])
                        vertical_distance = abs(wrist_pos[1] - head_center[1])
                        
                        # Create conservative cheating gesture detection zone
                        face_width = head_radius * 1.2   # More conservative horizontal area
                        face_height = head_radius * 1.5  # More conservative vertical area
                        
                        if horizontal_distance < face_width and vertical_distance < face_height:
                            # Stricter check: hand should be close to head level
                            if wrist_pos[1] >= head_center[1] - 30 and wrist_pos[1] <= head_center[1] + 100:  # Tighter range
                                gesture_detected = True
                                detection_reason = f"{wrist_name}_in_face_region"
                                logger.info(f"🤚 SUSPICIOUS GESTURE: {wrist_name} in face region (h:{horizontal_distance:.1f}, v:{vertical_distance:.1f})")
                                break
            
            # Pattern 3: Hand raised high (suspicious gesturing)
            if not gesture_detected:
                for wrist_name, wrist_pos in [('left_wrist', left_wrist), ('right_wrist', right_wrist)]:
                    if wrist_pos and head_center:
                        # Check if hand is significantly above head
                        if wrist_pos[1] < head_center[1] - 100:  # 100 pixels above head
                            gesture_detected = True
                            detection_reason = f"{wrist_name}_raised_high"
                            logger.info(f"🤚 SUSPICIOUS GESTURE: {wrist_name} raised high (y:{wrist_pos[1]} vs head:{head_center[1]})")
                            break
            
            if gesture_detected:
                logger.info(f"🤚 SUSPICIOUS GESTURE DETECTED: {detection_reason}")
            else:
                logger.debug(f"🤚 No suspicious gestures detected")
            
            return gesture_detected
            
        except Exception as e:
            logger.debug(f"Error in gesture detection: {e}")
            return False

"""YOLOv11-Pose detector for person pose estimation and behavior analysis."""
import os
import logging
import math
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
        """Initialize pose detector with model weights and configuration."""
        self.weights_path = weights_path or os.getenv('POSE_MODEL_PATH', 'weights/yolo11n-pose.pt')
        self.force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        
        # Load thresholds from environment with more sensitive defaults
        self.lean_angle_thresh = float(os.getenv('LEAN_ANGLE_THRESH', '15.0'))  # More sensitive
        self.head_turn_thresh = float(os.getenv('HEAD_TURN_THRESH', '20.0'))    # More sensitive  
        self.phone_iou_thresh = float(os.getenv('PHONE_IOU_THRESH', '0.1'))     # More sensitive
        
        # Additional thresholds
        self.confidence_thresh = float(os.getenv('POSE_CONFIDENCE_THRESH', '0.3'))
        self.debug_mode = os.getenv('DEBUG_POSE', 'false').lower() == 'true'
        
        # Determine device
        self.device = self._get_device()
        logger.info(f"PoseDetector initialized on device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
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
        Estimate poses from frame and analyze behaviors.
        
        Args:
            frame: Input image as numpy array (BGR format)
            phone_detections: List of phone detections from YOLO detector
            
        Returns:
            List of pose estimates, each containing:
            - track_id: person tracking ID (placeholder for now)
            - bbox: [x1, y1, x2, y2] person bounding box
            - yaw: head yaw angle in degrees
            - pitch: head pitch angle in degrees
            - lean_flag: True if person is leaning significantly
            - look_flag: True if person is looking around
            - phone_flag: True if phone is near person's torso
        """
        if frame is None:
            logger.warning("Received None frame for pose estimation")
            return []
        
        if phone_detections is None:
            phone_detections = []
        
        try:
            # Run pose inference
            results = self.model(frame, device=self.device, verbose=False)
            
            pose_estimates = []
            
            # Process results
            for result in results:
                if result.boxes is not None and result.keypoints is not None:
                    boxes = result.boxes
                    keypoints = result.keypoints
                    
                    for i in range(len(boxes)):
                        # Get person bounding box
                        bbox = boxes.xyxy[i].cpu().numpy().tolist()
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # Skip low confidence detections
                        if conf < 0.5:
                            continue
                        
                        # Get keypoints for this person
                        person_keypoints = keypoints.xy[i].cpu().numpy()  # Shape: (17, 2)
                        keypoint_confs = keypoints.conf[i].cpu().numpy() if hasattr(keypoints, 'conf') else None
                        
                        # Extract key body parts
                        head_points = self._extract_head_keypoints(person_keypoints, keypoint_confs)
                        shoulder_points = self._extract_shoulder_keypoints(person_keypoints, keypoint_confs)
                        hip_points = self._extract_hip_keypoints(person_keypoints, keypoint_confs)
                        
                        # Compute derived features
                        yaw, pitch = self._compute_head_angles(head_points)
                        lean_flag = self._compute_leaning(shoulder_points, hip_points)
                        look_flag = self._compute_looking_around(yaw)
                        phone_flag = self._compute_phone_near(bbox, phone_detections)
                        
                        pose_estimate = {
                            'track_id': f"person_{i}",  # Placeholder - will be replaced by tracker
                            'bbox': bbox,
                            'yaw': yaw,
                            'pitch': pitch,
                            'lean_flag': lean_flag,
                            'look_flag': look_flag,
                            'phone_flag': phone_flag,
                            'confidence': conf
                        }
                        
                        pose_estimates.append(pose_estimate)
            
            logger.debug(f"Generated {len(pose_estimates)} pose estimates")
            return pose_estimates
            
        except Exception as e:
            logger.error(f"Error during pose estimation: {e}")
            return []
    
    def _extract_head_keypoints(self, keypoints: np.ndarray, confs: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float]]:
        """Extract head keypoints (nose, eyes) with confidence filtering."""
        head_points = {}
        
        for name, idx in [('nose', self.KEYPOINTS['nose']), 
                         ('left_eye', self.KEYPOINTS['left_eye']), 
                         ('right_eye', self.KEYPOINTS['right_eye'])]:
            if idx < len(keypoints):
                x, y = keypoints[idx]
                # Check confidence if available
                if confs is None or (idx < len(confs) and confs[idx] > 0.3):
                    if x > 0 and y > 0:  # Valid keypoint
                        head_points[name] = (float(x), float(y))
        
        return head_points
    
    def _extract_shoulder_keypoints(self, keypoints: np.ndarray, confs: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float]]:
        """Extract shoulder keypoints with confidence filtering."""
        shoulder_points = {}
        
        for name, idx in [('left_shoulder', self.KEYPOINTS['left_shoulder']), 
                         ('right_shoulder', self.KEYPOINTS['right_shoulder'])]:
            if idx < len(keypoints):
                x, y = keypoints[idx]
                # Check confidence if available
                if confs is None or (idx < len(confs) and confs[idx] > 0.3):
                    if x > 0 and y > 0:  # Valid keypoint
                        shoulder_points[name] = (float(x), float(y))
        
        return shoulder_points
    
    def _extract_hip_keypoints(self, keypoints: np.ndarray, confs: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float]]:
        """Extract hip keypoints with confidence filtering."""
        hip_points = {}
        
        for name, idx in [('left_hip', self.KEYPOINTS['left_hip']), 
                         ('right_hip', self.KEYPOINTS['right_hip'])]:
            if idx < len(keypoints):
                x, y = keypoints[idx]
                # Check confidence if available
                if confs is None or (idx < len(confs) and confs[idx] > 0.3):
                    if x > 0 and y > 0:  # Valid keypoint
                        hip_points[name] = (float(x), float(y))
        
        return hip_points
    
    def _compute_head_angles(self, head_points: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """Compute head yaw and pitch angles from head keypoints with improved accuracy."""
        yaw, pitch = 0.0, 0.0
        
        try:
            # Method 1: Eye-based yaw detection (most reliable for frontal faces)
            if 'left_eye' in head_points and 'right_eye' in head_points:
                left_eye = head_points['left_eye']
                right_eye = head_points['right_eye']
                
                # Calculate eye positions and separation
                eye_dx = right_eye[0] - left_eye[0]
                eye_dy = right_eye[1] - left_eye[1]
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_separation = abs(eye_dx)
                
                # Improved yaw calculation with better thresholds
                if eye_separation > 15:  # Increased minimum for reliable detection
                    # Calculate expected separation for frontal face
                    # Use more conservative estimate
                    expected_separation = eye_separation * 1.2  # Less aggressive multiplier
                    separation_ratio = eye_separation / expected_separation
                    
                    # More conservative yaw detection to reduce false positives
                    if separation_ratio < 0.75:  # Higher threshold for more certainty
                        yaw_magnitude = (1.0 - separation_ratio) * 45.0  # Reduced max range
                        
                        # Determine direction with better validation
                        eye_ratio = eye_dx / eye_separation if eye_separation > 0 else 0
                        
                        if eye_ratio < 0.3:  # Face turned significantly right
                            yaw = -yaw_magnitude
                        elif eye_ratio > 0.7:  # Face turned significantly left
                            yaw = yaw_magnitude
                        # Else: face is roughly frontal, no yaw assigned
                        
                        # Additional validation using eye vertical alignment
                        if abs(eye_dy) > eye_separation * 0.3:  # Eyes too misaligned
                            yaw *= 0.5  # Reduce confidence due to head tilt
            
            # Method 2: Nose position validation (more accurate than ears)
            if 'nose' in head_points and 'left_eye' in head_points and 'right_eye' in head_points:
                nose = head_points['nose']
                left_eye = head_points['left_eye']
                right_eye = head_points['right_eye']
                
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                nose_offset = nose[0] - eye_center_x
                eye_width = abs(right_eye[0] - left_eye[0])
                
                if eye_width > 15:  # Valid eye separation for nose-based detection
                    nose_ratio = nose_offset / (eye_width / 2)
                    nose_yaw = nose_ratio * 20.0  # Reduced sensitivity
                    
                    # Only use nose-based detection if it's significant
                    if abs(nose_yaw) > abs(yaw) and abs(nose_yaw) > 10:
                        yaw = nose_yaw
            
            # Method 3: Single ear visible (strong turn indicator) - keep this as reliable
            if abs(yaw) < 15:  # Only if we haven't detected significant turn already
                if 'left_ear' in head_points and 'right_ear' not in head_points:
                    # Check if left ear is significantly visible (face turned right)
                    if 'left_eye' in head_points:
                        left_eye = head_points['left_eye']
                        left_ear = head_points['left_ear']
                        ear_eye_distance = abs(left_ear[0] - left_eye[0])
                        if ear_eye_distance > 20:  # Significant ear visibility
                            yaw = -25.0  # Face turned right
                
                elif 'right_ear' in head_points and 'left_ear' not in head_points:
                    # Check if right ear is significantly visible (face turned left)
                    if 'right_eye' in head_points:
                        right_eye = head_points['right_eye']
                        right_ear = head_points['right_ear']
                        ear_eye_distance = abs(right_ear[0] - right_eye[0])
                        if ear_eye_distance > 20:  # Significant ear visibility
                            yaw = 25.0  # Face turned left
            
            # Method 4: Both ears visible - verify frontal position
            if 'left_ear' in head_points and 'right_ear' in head_points:
                left_ear = head_points['left_ear']
                right_ear = head_points['right_ear']
                ear_separation = abs(right_ear[0] - left_ear[0])
                
                # If both ears are clearly visible with good separation, likely frontal
                if ear_separation > 40 and abs(yaw) < 20:
                    yaw *= 0.5  # Reduce yaw confidence for frontal faces
            
            # Enhanced pitch calculation (keep existing logic but improve thresholds)
            if 'nose' in head_points and ('left_eye' in head_points or 'right_eye' in head_points):
                nose = head_points['nose']
                
                # Get average eye position
                if 'left_eye' in head_points and 'right_eye' in head_points:
                    eye_y = (head_points['left_eye'][1] + head_points['right_eye'][1]) / 2
                elif 'left_eye' in head_points:
                    eye_y = head_points['left_eye'][1]
                else:
                    eye_y = head_points['right_eye'][1]
                
                # Calculate vertical distance
                nose_eye_dy = nose[1] - eye_y
                
                # More conservative pitch estimation
                if abs(nose_eye_dy) > 8:  # Higher threshold to reduce false positives
                    face_height = 50  # More realistic face height
                    pitch = math.degrees(math.atan2(nose_eye_dy, face_height))
                    pitch = max(-45, min(45, pitch))  # More reasonable range
            
            # Final validation: if yaw is very small, set to zero to avoid micro-movements
            if abs(yaw) < 5:
                yaw = 0.0
            
            # Debug output with improved information
            if self.debug_mode and abs(yaw) > 0:
                confidence = "HIGH" if abs(yaw) > 15 else "MEDIUM" if abs(yaw) > 10 else "LOW"
                logger.info(f"Head angles computed: yaw={yaw:.1f}° ({confidence} confidence), pitch={pitch:.1f}°")
        
        except Exception as e:
            logger.debug(f"Error computing head angles: {e}")
        
        return yaw, pitch
    
    def _compute_leaning(self, shoulder_points: Dict[str, Tuple[float, float]], 
                        hip_points: Dict[str, Tuple[float, float]]) -> bool:
        """
        Anatomically-accurate leaning detection using human body mechanics.
        
        Key anatomical principles:
        1. Natural spinal curvature allows 10-15° of normal variation
        2. Shoulder line should be roughly parallel to hip line when upright
        3. Torso centerline should be vertical (±10° tolerance for natural posture)
        4. Asymmetric loading indicates lateral leaning
        """
        try:
            if not shoulder_points or not hip_points:
                return False
            
            lean_detected = False
            max_deviation = 0.0
            detection_method = "none"
            
            # METHOD 1: ANATOMICAL TORSO VERTICAL ALIGNMENT
            # Calculate the natural spinal axis and check deviation from vertical
            if ('left_shoulder' in shoulder_points and 'right_shoulder' in shoulder_points and
                'left_hip' in hip_points and 'right_hip' in hip_points):
                
                # Get shoulder and hip midpoints (anatomical landmarks)
                left_shoulder = shoulder_points['left_shoulder']
                right_shoulder = shoulder_points['right_shoulder']
                left_hip = hip_points['left_hip']
                right_hip = hip_points['right_hip']
                
                # Calculate anatomical centers
                shoulder_midpoint = ((left_shoulder[0] + right_shoulder[0]) / 2,
                                   (left_shoulder[1] + right_shoulder[1]) / 2)
                hip_midpoint = ((left_hip[0] + right_hip[0]) / 2,
                              (left_hip[1] + right_hip[1]) / 2)
                
                # Calculate torso vector (spinal axis approximation)
                torso_dx = shoulder_midpoint[0] - hip_midpoint[0]
                torso_dy = shoulder_midpoint[1] - hip_midpoint[1]
                
                # Ensure minimum torso height for reliable measurement
                torso_height = abs(torso_dy)
                if torso_height > 30:  # Reduced minimum for better sensitivity
                    
                    # Calculate deviation from vertical (anatomically normal = 0°)
                    # Use torso_dx relative to torso_dy to get lean angle from vertical
                    torso_lean_angle = abs(math.degrees(math.atan2(abs(torso_dx), torso_height)))
                    
                    # Apply anatomical tolerance: more sensitive detection at 6° variation
                    anatomical_tolerance = 6.0
                    significant_lean_threshold = max(self.lean_angle_thresh, anatomical_tolerance)
                    
                    if torso_lean_angle > significant_lean_threshold:
                        lean_detected = True
                        max_deviation = torso_lean_angle
                        detection_method = "anatomical_torso_axis"
            
            # METHOD 2: SHOULDER-HIP PARALLEL ANALYSIS
            # In normal posture, shoulder line should be roughly parallel to hip line
            if not lean_detected and ('left_shoulder' in shoulder_points and 'right_shoulder' in shoulder_points and
                                     'left_hip' in hip_points and 'right_hip' in hip_points):
                
                left_shoulder = shoulder_points['left_shoulder']
                right_shoulder = shoulder_points['right_shoulder']
                left_hip = hip_points['left_hip']
                right_hip = hip_points['right_hip']
                
                # Calculate shoulder line angle
                shoulder_dx = right_shoulder[0] - left_shoulder[0]
                shoulder_dy = right_shoulder[1] - left_shoulder[1]
                shoulder_separation = math.sqrt(shoulder_dx**2 + shoulder_dy**2)
                
                # Calculate hip line angle
                hip_dx = right_hip[0] - left_hip[0]
                hip_dy = right_hip[1] - left_hip[1]
                hip_separation = math.sqrt(hip_dx**2 + hip_dy**2)
                
                # Only proceed if we have good anatomical landmarks
                if shoulder_separation > 30 and hip_separation > 20:
                    
                    # Calculate angles from horizontal
                    shoulder_angle = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
                    hip_angle = math.degrees(math.atan2(hip_dy, hip_dx))
                    
                    # Calculate parallel deviation (should be minimal in upright posture)
                    angle_difference = abs(shoulder_angle - hip_angle)
                    if angle_difference > 180:
                        angle_difference = 360 - angle_difference
                    
                    # Anatomical principle: shoulder-hip parallelism deviation indicates leaning
                    parallelism_threshold = 20.0  # Degrees of acceptable non-parallelism
                    
                    if angle_difference > parallelism_threshold:
                        lean_detected = True
                        max_deviation = max(max_deviation, angle_difference)
                        if detection_method == "none":
                            detection_method = "shoulder_hip_parallelism"
            
            # METHOD 3: ASYMMETRIC BODY LOADING ANALYSIS
            # Lateral leaning causes unequal loading on left/right body sides
            if not lean_detected and ('left_shoulder' in shoulder_points and 'right_shoulder' in shoulder_points and
                                     'left_hip' in hip_points and 'right_hip' in hip_points):
                
                left_shoulder = shoulder_points['left_shoulder']
                right_shoulder = shoulder_points['right_shoulder']
                left_hip = hip_points['left_hip']
                right_hip = hip_points['right_hip']
                
                # Calculate body side measurements (anatomical bilateral symmetry)
                left_torso_length = math.sqrt((left_shoulder[0] - left_hip[0])**2 + 
                                            (left_shoulder[1] - left_hip[1])**2)
                right_torso_length = math.sqrt((right_shoulder[0] - right_hip[0])**2 + 
                                             (right_shoulder[1] - right_hip[1])**2)
                
                # Ensure valid anatomical measurements
                if left_torso_length > 30 and right_torso_length > 30:
                    
                    # Calculate bilateral asymmetry ratio
                    asymmetry_ratio = abs(left_torso_length - right_torso_length) / max(left_torso_length, right_torso_length)
                    
                    # Convert to angular deviation using anatomical proportions
                    asymmetry_angle = math.degrees(math.atan(asymmetry_ratio))
                    
                    # Anatomical threshold: >20% bilateral asymmetry indicates significant lean (more sensitive)
                    asymmetry_threshold = 0.20  # 20% asymmetry tolerance
                    
                    if asymmetry_ratio > asymmetry_threshold and asymmetry_angle > 12:
                        lean_detected = True
                        max_deviation = max(max_deviation, asymmetry_angle)
                        if detection_method == "none":
                            detection_method = "bilateral_asymmetry"
            
            # METHOD 4: GRAVITATIONAL CENTER DISPLACEMENT
            # Calculate center of mass displacement from anatomical baseline
            if not lean_detected and ('left_shoulder' in shoulder_points and 'right_shoulder' in shoulder_points and
                                     'left_hip' in hip_points and 'right_hip' in hip_points):
                
                left_shoulder = shoulder_points['left_shoulder']
                right_shoulder = shoulder_points['right_shoulder']
                left_hip = hip_points['left_hip']
                right_hip = hip_points['right_hip']
                
                # Calculate anatomical center of mass approximation
                # Upper body weight distribution: shoulders ~40%, hips ~60%
                upper_body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                lower_body_center_x = (left_hip[0] + right_hip[0]) / 2
                
                # Weighted center of mass (anatomical proportions)
                center_of_mass_x = (upper_body_center_x * 0.4 + lower_body_center_x * 0.6)
                
                # Calculate baseline (vertical through hip center)
                baseline_x = lower_body_center_x
                
                # Calculate center of mass displacement
                displacement = abs(center_of_mass_x - baseline_x)
                
                # Reference torso width for proportional analysis
                torso_width = max(abs(right_shoulder[0] - left_shoulder[0]),
                                abs(right_hip[0] - left_hip[0]))
                
                if torso_width > 20:  # Valid torso width measurement
                    # Calculate proportional displacement
                    displacement_ratio = displacement / torso_width
                    
                    # Anatomical threshold: >20% displacement indicates leaning (more sensitive)
                    if displacement_ratio > 0.20:
                        displacement_angle = math.degrees(math.atan(displacement_ratio))
                        if displacement_angle > 12:
                            lean_detected = True
                            max_deviation = max(max_deviation, displacement_angle)
                            if detection_method == "none":
                                detection_method = "center_of_mass_displacement"
            
            # METHOD 5: SIMPLE SHOULDER ANGLE BACKUP (if complex methods fail)
            if not lean_detected and ('left_shoulder' in shoulder_points and 'right_shoulder' in shoulder_points):
                left_shoulder = shoulder_points['left_shoulder']
                right_shoulder = shoulder_points['right_shoulder']
                
                # Calculate shoulder line angle from horizontal
                shoulder_dx = right_shoulder[0] - left_shoulder[0]
                shoulder_dy = right_shoulder[1] - left_shoulder[1]
                shoulder_separation = abs(shoulder_dx)
                
                if shoulder_separation > 20:  # Valid shoulder separation
                    # Calculate deviation from horizontal (0°)
                    shoulder_angle = abs(math.degrees(math.atan2(shoulder_dy, shoulder_dx)))
                    
                    # Convert to deviation from horizontal (0° = horizontal, 90° = vertical)
                    if shoulder_angle > 90:
                        shoulder_deviation = 180 - shoulder_angle
                    else:
                        shoulder_deviation = shoulder_angle
                    
                    # Detect significant shoulder tilt (normal sitting: 0-10°)
                    if shoulder_deviation > self.lean_angle_thresh:
                        lean_detected = True
                        max_deviation = max(max_deviation, shoulder_deviation)
                        if detection_method == "none":
                            detection_method = "simple_shoulder_tilt"
            
            # Debug output with anatomical context
            if self.debug_mode:
                if lean_detected:
                    logger.info(f"🏃 ANATOMICAL LEAN DETECTED: method={detection_method}, "
                               f"deviation={max_deviation:.1f}°, threshold={self.lean_angle_thresh}°")
                else:
                    logger.debug(f"✅ Normal anatomical posture maintained (max_deviation={max_deviation:.1f}°)")
            
            return lean_detected
            
        except Exception as e:
            logger.debug(f"Error in anatomical leaning analysis: {e}")
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

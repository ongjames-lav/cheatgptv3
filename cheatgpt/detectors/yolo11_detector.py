"""YOLOv11 detector for person and cell phone detection."""
import os
import logging
from typing import List, Dict, Any
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLO11Detector:
    """YOLOv11 detector focused on person and cell phone detection."""
    
    # Target classes we want to detect
    TARGET_CLASSES = {'person', 'cell phone'}
    
    def __init__(self, weights_path=None):
        """Initialize YOLO11 detector with model weights and device configuration."""
        self.weights_path = weights_path or os.getenv('YOLO_MODEL_PATH', 'weights/yolo11m.pt')
        self.force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        
        # Determine device
        self.device = self._get_device()
        logger.info(f"YOLO11Detector initialized on device: {self.device}")
        
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
        """Load the YOLO model with the specified weights."""
        try:
            # Convert relative path to absolute if needed
            if not os.path.isabs(self.weights_path):
                base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to cheatgpt dir
                self.weights_path = os.path.join(base_dir, self.weights_path)
            
            logger.info(f"Loading YOLO model from: {self.weights_path}")
            model = YOLO(self.weights_path)
            
            # Move model to the appropriate device
            model.to(self.device)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons and cell phones in the given frame with improved filtering.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2] in xyxy format
            - conf: confidence score (0-1)
            - cls_name: class name ('person' or 'cell phone')
        """
        if frame is None:
            logger.warning("Received None frame for detection")
            return []
        
        try:
            # Run inference with improved settings
            results = self.model(frame, device=self.device, verbose=False, 
                               conf=0.3, iou=0.5, max_det=50)
            
            detections = []
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    # Get class names
                    class_names = result.names
                    
                    for i in range(len(boxes)):
                        # Get class name
                        cls_id = int(boxes.cls[i])
                        cls_name = class_names[cls_id]
                        
                        # Filter for target classes only
                        if cls_name in self.TARGET_CLASSES:
                            # Get bounding box in xyxy format
                            bbox = boxes.xyxy[i].cpu().numpy().tolist()
                            
                            # Get confidence score
                            conf = float(boxes.conf[i].cpu().numpy())
                            
                            # Apply additional confidence filtering
                            min_conf = 0.4 if cls_name == 'person' else 0.3  # Higher threshold for persons
                            if conf >= min_conf:
                                detection = {
                                    'bbox': bbox,  # [x1, y1, x2, y2]
                                    'conf': conf,
                                    'cls_name': cls_name
                                }
                                
                                detections.append(detection)
            
            # Apply additional NMS to remove duplicates
            detections = self._apply_class_nms(detections)
            
            logger.debug(f"Found {len(detections)} filtered detections")
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def _apply_class_nms(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression within each class to remove duplicates."""
        if not detections:
            return detections
        
        # Group detections by class
        class_groups = {}
        for det in detections:
            cls_name = det['cls_name']
            if cls_name not in class_groups:
                class_groups[cls_name] = []
            class_groups[cls_name].append(det)
        
        filtered_detections = []
        
        # Apply NMS for each class
        for cls_name, cls_detections in class_groups.items():
            if len(cls_detections) <= 1:
                filtered_detections.extend(cls_detections)
                continue
            
            # Sort by confidence (highest first)
            cls_detections.sort(key=lambda x: x['conf'], reverse=True)
            
            # Apply NMS
            keep_indices = []
            for i, det_i in enumerate(cls_detections):
                keep = True
                for j in keep_indices:
                    det_j = cls_detections[j]
                    iou = self._calculate_iou(det_i['bbox'], det_j['bbox'])
                    if iou > iou_threshold:
                        keep = False
                        break
                if keep:
                    keep_indices.append(i)
            
            # Keep only non-suppressed detections
            for i in keep_indices:
                filtered_detections.append(cls_detections[i])
        
        return filtered_detections
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': self.weights_path,
            'device': self.device,
            'target_classes': list(self.TARGET_CLASSES),
            'cuda_available': torch.cuda.is_available(),
            'force_cpu': self.force_cpu
        }

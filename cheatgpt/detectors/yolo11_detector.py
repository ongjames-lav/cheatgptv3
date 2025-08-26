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
        self.weights_path = weights_path or os.getenv('YOLO_MODEL_PATH', 'weights/yolo11n.pt')
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
        Detect persons and cell phones in the given frame.
        
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
            # Run inference
            results = self.model(frame, device=self.device, verbose=False)
            
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
                            
                            detection = {
                                'bbox': bbox,  # [x1, y1, x2, y2]
                                'conf': conf,
                                'cls_name': cls_name
                            }
                            
                            detections.append(detection)
            
            logger.debug(f"Found {len(detections)} detections")
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': self.weights_path,
            'device': self.device,
            'target_classes': list(self.TARGET_CLASSES),
            'cuda_available': torch.cuda.is_available(),
            'force_cpu': self.force_cpu
        }

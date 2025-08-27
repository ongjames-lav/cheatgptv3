#!/usr/bin/env python3
"""
Enhanced CheatGPT Engine Integration
Combines realistic LSTM (88.64% accuracy) with COCO-enhanced rule-based detection
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our components
from realistic_lstm_integration import RealisticLSTMClassifier

# Try to import existing engine components
try:
    from cheatgpt.pose_detector import PoseDetector
    from cheatgpt.engine import Engine
    CHEATGPT_AVAILABLE = True
except ImportError:
    logger.warning("CheatGPT engine not available, using mock components")
    CHEATGPT_AVAILABLE = False
    
    # Mock components for testing
    class PoseDetector:
        def detect_poses(self, frame):
            return {'poses': [{'bbox': [100, 100, 200, 300], 'keypoints': {}}]}
    
    class Engine:
        def _detect_behavior_for_person(self, frame, pose_data):
            return {
                'gesture_flag': 0, 'look_flag': 0, 'lean_flag': 0,
                'confidence': 0.5, 'lean_angle': 0, 'head_turn_angle': 0
            }

class EnhancedCheatGPTEngine:
    """Enhanced CheatGPT Engine with realistic LSTM and COCO integration"""
    
    def __init__(self, use_enhanced_lstm: bool = True):
        
        # Initialize components
        self.pose_detector = PoseDetector()
        self.base_engine = Engine()
        
        # Enhanced LSTM integration
        self.use_enhanced_lstm = use_enhanced_lstm
        self.enhanced_lstm = None
        self.detection_history = []
        self.prediction_cache = {}
        
        # COCO-enhanced configuration
        self.coco_config = self._load_coco_config()
        
        # Performance tracking
        self.frame_count = 0
        self.detection_stats = {
            'total_detections': 0,
            'enhanced_predictions': 0,
            'fallback_predictions': 0,
            'average_confidence': 0.0
        }
        
        # Initialize enhanced LSTM
        if use_enhanced_lstm:
            self._initialize_enhanced_lstm()
        
        logger.info("🚀 Enhanced CheatGPT Engine initialized")
    
    def _load_coco_config(self) -> Dict:
        """Load COCO-enhanced configuration"""
        try:
            with open("config/enhanced/coco_enhanced_config.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("COCO config not found, using defaults")
            return self._get_default_coco_config()
    
    def _get_default_coco_config(self) -> Dict:
        """Default COCO configuration"""
        return {
            'pose_detection': {
                'gesture_detection': {
                    'min_confidence': 0.35,
                    'spatial_tolerance': 0.15
                },
                'looking_detection': {
                    'min_confidence': 0.4,
                    'spatial_tolerance': 0.12
                }
            },
            'behavioral_analysis': {
                'gesture_persistence_frames': 5,
                'looking_persistence_frames': 7,
                'confidence_accumulation': True
            }
        }
    
    def _initialize_enhanced_lstm(self):
        """Initialize the realistic LSTM classifier"""
        try:
            self.enhanced_lstm = RealisticLSTMClassifier()
            if self.enhanced_lstm.is_loaded:
                logger.info("✅ Enhanced LSTM loaded (88.64% accuracy)")
            else:
                logger.warning("⚠️ Enhanced LSTM not available, using fallback")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced LSTM: {e}")
            self.enhanced_lstm = None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame with enhanced detection"""
        
        self.frame_count += 1
        start_time = time.time()
        
        # Base pose detection
        pose_results = self.pose_detector.detect_poses(frame)
        
        if not pose_results.get('poses'):
            return self._create_empty_result(frame)
        
        # Enhanced detection for each person
        enhanced_results = []
        for pose_data in pose_results['poses']:
            enhanced_detection = self._process_person_detection(frame, pose_data)
            enhanced_results.append(enhanced_detection)
        
        # Combine results
        final_result = self._combine_detection_results(enhanced_results, frame)
        
        # Performance tracking
        processing_time = time.time() - start_time
        final_result['processing_time'] = processing_time
        final_result['frame_number'] = self.frame_count
        
        return final_result
    
    def _process_person_detection(self, frame: np.ndarray, pose_data: Dict) -> Dict:
        """Process detection for a single person with COCO enhancement"""
        
        # Base rule-based detection
        base_detection = self._enhanced_rule_based_detection(frame, pose_data)
        
        # Add to detection history
        self.detection_history.append(base_detection)
        if len(self.detection_history) > 15:  # Keep last 15 detections
            self.detection_history.pop(0)
        
        # Enhanced LSTM prediction
        enhanced_prediction = None
        if self.enhanced_lstm and len(self.detection_history) >= 5:
            enhanced_prediction = self.enhanced_lstm.predict_behavior_sequence(
                self.detection_history[-10:]  # Use last 10 detections
            )
        
        # Combine base detection with enhanced prediction
        combined_result = self._combine_base_and_enhanced(base_detection, enhanced_prediction)
        
        # Update statistics
        self._update_detection_stats(combined_result)
        
        return combined_result
    
    def _enhanced_rule_based_detection(self, frame: np.ndarray, pose_data: Dict) -> Dict:
        """Enhanced rule-based detection using COCO spatial patterns"""
        
        # Get base detection from original engine
        base_result = self.base_engine._detect_behavior_for_person(frame, pose_data)
        
        # COCO spatial enhancement
        if 'bbox' in pose_data:
            bbox = pose_data['bbox']
            frame_h, frame_w = frame.shape[:2]
            
            # Calculate enhanced spatial features
            center_x = (bbox[0] + bbox[2]/2) / frame_w
            center_y = (bbox[1] + bbox[3]/2) / frame_h
            bbox_area = (bbox[2] * bbox[3]) / (frame_w * frame_h)
            
            # Apply COCO-based confidence adjustments
            confidence_adjustment = self._calculate_coco_confidence_adjustment(
                base_result, center_x, center_y, bbox_area
            )
            
            # Update base result with enhancements
            base_result['confidence'] *= confidence_adjustment
            base_result['center_x'] = center_x
            base_result['center_y'] = center_y
            base_result['bbox_area'] = bbox_area
            base_result['coco_enhanced'] = True
        
        return base_result
    
    def _calculate_coco_confidence_adjustment(self, detection: Dict, 
                                            center_x: float, center_y: float, 
                                            bbox_area: float) -> float:
        """Calculate confidence adjustment based on COCO spatial patterns"""
        
        adjustment = 1.0
        
        # Gesture detection enhancement
        if detection.get('gesture_flag', 0):
            gesture_config = self.coco_config['pose_detection']['gesture_detection']
            
            # Check if spatial properties match expected gesture patterns
            if 0.3 <= center_x <= 0.7 and 0.2 <= center_y <= 0.6:  # Typical gesture region
                adjustment *= 1.15  # Boost confidence
            
            if 0.1 <= bbox_area <= 0.7:  # Reasonable gesture size
                adjustment *= 1.1
        
        # Looking detection enhancement
        if detection.get('look_flag', 0):
            looking_config = self.coco_config['pose_detection']['looking_detection']
            
            # Head region focus for looking behaviors
            if 0.25 <= center_y <= 0.5:  # Head region
                adjustment *= 1.2
            
            # Reasonable head bbox size
            if 0.05 <= bbox_area <= 0.4:
                adjustment *= 1.1
        
        # Normal behavior validation
        if not detection.get('gesture_flag', 0) and not detection.get('look_flag', 0):
            # Stable, centered poses get confidence boost
            center_stability = 1.0 - abs(center_x - 0.5) - abs(center_y - 0.45)
            if center_stability > 0.3:
                adjustment *= (1.0 + center_stability * 0.1)
        
        return min(adjustment, 1.5)  # Cap adjustment at 1.5x
    
    def _combine_base_and_enhanced(self, base_detection: Dict, 
                                 enhanced_prediction: Optional[Dict]) -> Dict:
        """Combine base rule-based detection with enhanced LSTM prediction"""
        
        result = base_detection.copy()
        
        if enhanced_prediction:
            # Enhanced prediction available
            result['enhanced_prediction'] = enhanced_prediction['prediction']
            result['enhanced_confidence'] = enhanced_prediction['confidence']
            result['lstm_probabilities'] = enhanced_prediction.get('probabilities', {})
            result['auxiliary_predictions'] = enhanced_prediction.get('auxiliary_predictions', {})
            result['model_accuracy'] = 88.64
            
            # Blend base and enhanced predictions
            base_confidence = result.get('confidence', 0.5)
            enhanced_confidence = enhanced_prediction['confidence']
            
            # Weighted combination (70% enhanced, 30% base)
            combined_confidence = 0.7 * enhanced_confidence + 0.3 * base_confidence
            
            # Use enhanced prediction if confidence is high enough
            if enhanced_confidence > 0.6:
                result['final_prediction'] = enhanced_prediction['prediction']
                result['final_confidence'] = combined_confidence
                result['prediction_source'] = 'enhanced_lstm'
            else:
                result['final_prediction'] = self._map_base_to_enhanced_prediction(base_detection)
                result['final_confidence'] = base_confidence
                result['prediction_source'] = 'rule_based'
        else:
            # Fallback to base detection
            result['final_prediction'] = self._map_base_to_enhanced_prediction(base_detection)
            result['final_confidence'] = result.get('confidence', 0.5)
            result['prediction_source'] = 'rule_based_only'
            result['model_accuracy'] = None
        
        return result
    
    def _map_base_to_enhanced_prediction(self, base_detection: Dict) -> str:
        """Map base detection flags to enhanced prediction categories"""
        
        if base_detection.get('gesture_flag', 0):
            return 'suspicious_gesture'
        elif base_detection.get('look_flag', 0):
            return 'suspicious_looking'
        elif base_detection.get('lean_flag', 0):
            return 'suspicious_lean'
        else:
            return 'normal'
    
    def _combine_detection_results(self, person_results: List[Dict], 
                                 frame: np.ndarray) -> Dict:
        """Combine results from multiple people in the frame"""
        
        if not person_results:
            return self._create_empty_result(frame)
        
        # Find most suspicious detection
        most_suspicious = max(person_results, 
                            key=lambda x: x.get('final_confidence', 0))
        
        # Calculate overall frame assessment
        suspicious_count = sum(1 for result in person_results 
                             if result.get('final_prediction', 'normal') != 'normal')
        
        frame_result = {
            'timestamp': time.time(),
            'total_people': len(person_results),
            'suspicious_people': suspicious_count,
            'primary_detection': most_suspicious,
            'all_detections': person_results,
            'frame_assessment': self._assess_frame_risk(person_results),
            'enhanced_features': {
                'coco_integration': True,
                'realistic_lstm': self.enhanced_lstm is not None and self.enhanced_lstm.is_loaded,
                'spatial_enhancement': True
            }
        }
        
        return frame_result
    
    def _assess_frame_risk(self, person_results: List[Dict]) -> Dict:
        """Assess overall frame risk level"""
        
        if not person_results:
            return {'level': 'low', 'confidence': 0.0, 'reason': 'no_detections'}
        
        max_confidence = max(result.get('final_confidence', 0) for result in person_results)
        suspicious_behaviors = [result.get('final_prediction', 'normal') 
                              for result in person_results 
                              if result.get('final_prediction', 'normal') != 'normal']
        
        if max_confidence > 0.8 and suspicious_behaviors:
            level = 'high'
        elif max_confidence > 0.6 and suspicious_behaviors:
            level = 'medium'
        elif suspicious_behaviors:
            level = 'low'
        else:
            level = 'normal'
        
        return {
            'level': level,
            'confidence': max_confidence,
            'behaviors': suspicious_behaviors,
            'reasoning': f"{len(suspicious_behaviors)} suspicious behaviors detected"
        }
    
    def _create_empty_result(self, frame: np.ndarray) -> Dict:
        """Create empty result when no poses detected"""
        return {
            'timestamp': time.time(),
            'total_people': 0,
            'suspicious_people': 0,
            'primary_detection': None,
            'all_detections': [],
            'frame_assessment': {'level': 'normal', 'confidence': 0.0, 'reason': 'no_poses'},
            'enhanced_features': {
                'coco_integration': True,
                'realistic_lstm': self.enhanced_lstm is not None,
                'spatial_enhancement': True
            }
        }
    
    def _update_detection_stats(self, result: Dict):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += 1
        
        if result.get('prediction_source') == 'enhanced_lstm':
            self.detection_stats['enhanced_predictions'] += 1
        else:
            self.detection_stats['fallback_predictions'] += 1
        
        confidence = result.get('final_confidence', 0)
        total = self.detection_stats['total_detections']
        self.detection_stats['average_confidence'] = (
            (self.detection_stats['average_confidence'] * (total - 1) + confidence) / total
        )
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        info = {
            'engine_type': 'Enhanced CheatGPT Engine',
            'components': {
                'pose_detector': 'MediaPipe Pose',
                'rule_based': 'COCO-Enhanced Rules',
                'lstm_model': 'Realistic LSTM (88.64% accuracy)' if self.enhanced_lstm and self.enhanced_lstm.is_loaded else 'Not Available',
                'spatial_enhancement': 'COCO Spatial Patterns'
            },
            'performance': self.detection_stats,
            'frame_count': self.frame_count,
            'lstm_info': self.enhanced_lstm.get_model_info() if self.enhanced_lstm else None
        }
        return info
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.detection_stats = {
            'total_detections': 0,
            'enhanced_predictions': 0,
            'fallback_predictions': 0,
            'average_confidence': 0.0
        }
        self.frame_count = 0
        self.detection_history = []

if __name__ == "__main__":
    # Test the enhanced engine
    engine = EnhancedCheatGPTEngine()
    
    print("=== ENHANCED CHEATGPT ENGINE ===")
    info = engine.get_system_info()
    
    print(f"Engine: {info['engine_type']}")
    print("Components:")
    for component, details in info['components'].items():
        print(f"  {component}: {details}")
    
    if engine.enhanced_lstm and engine.enhanced_lstm.is_loaded:
        print("✅ Full enhanced system ready!")
        print("  - Realistic LSTM: 88.64% accuracy")
        print("  - COCO spatial enhancement")
        print("  - Enhanced rule-based detection")
    else:
        print("⚠️ Partial system (rule-based + COCO enhancement only)")
    
    print(f"\nLSTM Classes: {engine.enhanced_lstm.class_labels if engine.enhanced_lstm else 'Not available'}")

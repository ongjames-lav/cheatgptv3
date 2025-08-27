#!/usr/bin/env python3
"""
Enhanced Engine Integration for 6-Category Roboflow Dataset
Integrates the enhanced LSTM with rich behavioral features for superior detection
"""

import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from cheatgpt.temporal.lstm_model import LSTMClassifier
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLSTMClassifier(LSTMClassifier):
    """Enhanced LSTM classifier for 6-category behavior recognition"""
    
    def __init__(self, model_path: str = "weights/enhanced_lstm_behavior.pth",
                 label_encoder_path: str = "weights/enhanced_label_encoder.pkl"):
        # Initialize with enhanced model paths
        super().__init__(model_path, label_encoder_path)
        self.enhanced_features = True
        self.gesture_threshold = 0.7
        self.looking_threshold = 0.6
        
    def predict_enhanced(self, behavior_sequence: np.ndarray, 
                        additional_features: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced prediction with 6-category dataset features"""
        
        if not self.is_loaded:
            return self._get_default_result("Model not loaded")
        
        try:
            # Enhanced input validation for 12-dimensional features
            if isinstance(behavior_sequence, list):
                if len(behavior_sequence) == 0:
                    raise ValueError("Empty behavior sequence provided")
                behavior_sequence = np.array(behavior_sequence, dtype=np.float32)
            
            if behavior_sequence.size == 0:
                raise ValueError("Empty numpy array provided")
            
            # Validate enhanced feature dimensions (12D expected)
            if behavior_sequence.ndim == 2 and behavior_sequence.shape[1] != 12:
                logger.warning(f"Expected 12 features, got {behavior_sequence.shape[1]}. Adjusting...")
                behavior_sequence = self._adjust_feature_dimensions(behavior_sequence)
            elif behavior_sequence.ndim == 1:
                if len(behavior_sequence) != 12:
                    logger.warning(f"Expected 12 features, got {len(behavior_sequence)}. Adjusting...")
                behavior_sequence = self._adjust_feature_dimensions(behavior_sequence.reshape(1, -1))
                behavior_sequence = behavior_sequence.reshape(-1)
            
            # Enhanced preprocessing for 6-category features
            original_shape = behavior_sequence.shape
            target_seq_len = self.model.sequence_length
            target_input_size = 12  # Enhanced feature size
            
            # Handle sequence length
            if len(original_shape) == 1:
                behavior_sequence = np.tile(behavior_sequence.reshape(1, -1), (target_seq_len, 1))
            elif original_shape[0] < target_seq_len:
                if behavior_sequence.size > 0 and original_shape[0] > 0:
                    last_frame = behavior_sequence[-1:] 
                else:
                    last_frame = np.zeros((1, target_input_size), dtype=np.float32)
                padding_needed = target_seq_len - original_shape[0]
                padding = np.tile(last_frame, (padding_needed, 1))
                behavior_sequence = np.vstack([behavior_sequence, padding])
            elif original_shape[0] > target_seq_len:
                behavior_sequence = behavior_sequence[-target_seq_len:]
            
            # Enhanced feature validation
            if behavior_sequence.shape[1] != target_input_size:
                behavior_sequence = self._adjust_feature_dimensions(behavior_sequence)
            
            # Model inference
            input_tensor = torch.from_numpy(behavior_sequence).unsqueeze(0)
            if input_tensor.device != self.device:
                input_tensor = input_tensor.to(self.device, non_blocking=True)
            
            with torch.no_grad():
                self.model.eval()
                
                # Enhanced model output with auxiliary predictions
                if hasattr(self.model, 'forward') and 'main' in str(self.model.forward.__code__.co_varnames):
                    # Enhanced model with multi-task outputs
                    outputs = self.model(input_tensor)
                    main_output = outputs['main']
                    gesture_output = outputs.get('gesture', None)
                    looking_output = outputs.get('looking', None)
                    attention_weights = outputs.get('attention_weights', None)
                else:
                    # Fallback to regular model
                    main_output = self.model(input_tensor)
                    gesture_output = None
                    looking_output = None
                    attention_weights = None
                
                # Main prediction
                probabilities = torch.softmax(main_output, dim=1)
                predicted_class = int(torch.argmax(probabilities, dim=1).item())
                confidence = float(probabilities[0][predicted_class].item())
                
                # Enhanced confidence calculation
                max_prob = confidence
                if probabilities.shape[1] > 1:
                    second_max_prob = float(torch.topk(probabilities, min(2, probabilities.shape[1]), dim=1)[0][0][-1].item())
                    certainty_margin = max_prob - second_max_prob
                else:
                    certainty_margin = max_prob
                
                # Auxiliary predictions for enhanced analysis
                gesture_confidence = 0.0
                looking_confidence = 0.0
                
                if gesture_output is not None:
                    gesture_probs = torch.softmax(gesture_output, dim=1)
                    gesture_confidence = float(gesture_probs[0][1].item())  # Probability of gesture
                
                if looking_output is not None:
                    looking_probs = torch.softmax(looking_output, dim=1)
                    looking_confidence = float(looking_probs[0][1].item())  # Probability of looking
                
                # Enhanced label mapping
                if self.class_labels and 0 <= predicted_class < len(self.class_labels):
                    predicted_label = self.class_labels[predicted_class]
                else:
                    predicted_label = f"class_{predicted_class}"
                
                # Enhanced behavior analysis
                behavior_details = self._analyze_enhanced_behavior(
                    predicted_label, confidence, gesture_confidence, looking_confidence,
                    behavior_sequence, additional_features
                )
                
                # Enhanced result compilation
                result = {
                    'predicted_class': predicted_class,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'raw_confidence': confidence,
                    'certainty_margin': certainty_margin,
                    'probabilities': probabilities[0].cpu().numpy().tolist(),
                    'class_labels': self.class_labels,
                    'sequence_length_used': target_seq_len,
                    'input_features_used': target_input_size,
                    
                    # Enhanced auxiliary predictions
                    'gesture_confidence': gesture_confidence,
                    'looking_confidence': looking_confidence,
                    'gesture_detected': gesture_confidence > self.gesture_threshold,
                    'looking_detected': looking_confidence > self.looking_threshold,
                    
                    # Enhanced behavior analysis
                    'behavior_details': behavior_details,
                    'enhanced_features': True,
                    'attention_focused': attention_weights is not None,
                    
                    'error': None
                }
                
                return result
                
        except Exception as e:
            import traceback
            logger.error(f"Enhanced LSTM prediction error: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_default_result(str(e))
    
    def _adjust_feature_dimensions(self, features: np.ndarray) -> np.ndarray:
        """Adjust feature dimensions to match enhanced 12D format"""
        current_features = features.shape[1] if features.ndim == 2 else len(features)
        target_features = 12
        
        if current_features < target_features:
            # Pad with zeros for missing features
            if features.ndim == 1:
                padding = np.zeros(target_features - current_features, dtype=np.float32)
                return np.concatenate([features, padding])
            else:
                padding = np.zeros((features.shape[0], target_features - current_features), dtype=np.float32)
                return np.hstack([features, padding])
        elif current_features > target_features:
            # Truncate to target size
            return features[:, :target_features] if features.ndim == 2 else features[:target_features]
        
        return features
    
    def _analyze_enhanced_behavior(self, predicted_label: str, main_confidence: float,
                                 gesture_confidence: float, looking_confidence: float,
                                 sequence: np.ndarray, additional_features: Optional[Dict] = None) -> Dict:
        """Enhanced behavior analysis using 6-category dataset insights"""
        
        analysis = {
            'primary_behavior': predicted_label,
            'confidence_level': 'high' if main_confidence > 0.8 else 'medium' if main_confidence > 0.6 else 'low',
            'behavior_type': 'normal' if predicted_label == 'normal' else 'suspicious',
            'specific_indicators': []
        }
        
        # Gesture analysis
        if gesture_confidence > self.gesture_threshold:
            analysis['specific_indicators'].append({
                'type': 'hand_gesture',
                'confidence': gesture_confidence,
                'description': 'Suspicious hand movements detected (potential phone use or paper passing)'
            })
        
        # Looking behavior analysis
        if looking_confidence > self.looking_threshold:
            analysis['specific_indicators'].append({
                'type': 'head_turning',
                'confidence': looking_confidence,
                'description': 'Head turning behavior detected (potential looking at neighbors)'
            })
        
        # Sequence-based analysis
        if sequence.shape[0] >= 5:
            # Analyze temporal patterns
            gesture_trend = np.mean(sequence[:, 3])  # gesture_flag column
            looking_trend = np.mean(sequence[:, 1])  # look_flag column
            
            if gesture_trend > 0.3:
                analysis['specific_indicators'].append({
                    'type': 'persistent_gestures',
                    'confidence': gesture_trend,
                    'description': f'Persistent gesture activity over sequence ({gesture_trend:.1%})'
                })
            
            if looking_trend > 0.3:
                analysis['specific_indicators'].append({
                    'type': 'persistent_looking',
                    'confidence': looking_trend,
                    'description': f'Persistent looking behavior over sequence ({looking_trend:.1%})'
                })
        
        # Combined behavior analysis
        if gesture_confidence > 0.5 and looking_confidence > 0.5:
            analysis['specific_indicators'].append({
                'type': 'combined_suspicious',
                'confidence': (gesture_confidence + looking_confidence) / 2,
                'description': 'Multiple suspicious behaviors detected simultaneously'
            })
        
        # Additional feature analysis
        if additional_features:
            spatial_info = additional_features.get('spatial_features', {})
            if spatial_info.get('position_anomaly', False):
                analysis['specific_indicators'].append({
                    'type': 'position_anomaly',
                    'confidence': spatial_info.get('anomaly_score', 0.5),
                    'description': 'Unusual spatial positioning detected'
                })
        
        return analysis
    
    def _get_default_result(self, error_msg: str) -> Dict[str, Any]:
        """Get default enhanced result structure"""
        return {
            'predicted_class': 0,
            'predicted_label': 'normal',
            'confidence': 0.5,
            'raw_confidence': 0.5,
            'certainty_margin': 0.0,
            'probabilities': None,
            'class_labels': self.class_labels or ['normal', 'suspicious_gesture', 'suspicious_looking'],
            'sequence_length_used': 0,
            'input_features_used': 12,
            'gesture_confidence': 0.0,
            'looking_confidence': 0.0,
            'gesture_detected': False,
            'looking_detected': False,
            'behavior_details': {'primary_behavior': 'normal', 'confidence_level': 'low', 'behavior_type': 'normal', 'specific_indicators': []},
            'enhanced_features': True,
            'attention_focused': False,
            'error': error_msg
        }

def create_enhanced_behavior_vector(pose_data: Dict, additional_context: Optional[Dict] = None) -> np.ndarray:
    """Create enhanced 12-dimensional behavior vector from pose data"""
    
    # Extract basic rule-based features
    lean_flag = float(pose_data.get('lean_flag', 0))
    look_flag = float(pose_data.get('look_flag', 0))  
    phone_flag = float(pose_data.get('phone_flag', 0))
    
    # Enhanced gesture detection
    gesture_flag = float(pose_data.get('gesture_flag', 0))
    if not gesture_flag and phone_flag:  # Infer gesture from phone usage
        gesture_flag = 1.0
    
    # Extract continuous features
    lean_angle = pose_data.get('lean_angle', 0.0)
    head_turn_angle = pose_data.get('head_turn_angle', 0.0)
    confidence = pose_data.get('confidence', 0.7)
    
    # Extract spatial features
    bbox = pose_data.get('bbox', [0, 0, 100, 100])
    if len(bbox) >= 4:
        # Normalize bbox to image coordinates
        x, y, w, h = bbox[:4]
        center_x = (x + w/2) / 640  # Assuming 640x640 image size
        center_y = (y + h/2) / 640
        bbox_area = (w * h) / (640 * 640)
    else:
        center_x = center_y = bbox_area = 0.5
    
    # Create enhanced 12-dimensional feature vector
    enhanced_vector = np.array([
        # Basic rule-based flags (0-3)
        lean_flag,
        look_flag, 
        phone_flag,
        gesture_flag,
        
        # Normalized continuous features (4-6)
        np.clip(lean_angle / 45.0, -1.0, 1.0),
        np.clip(head_turn_angle / 90.0, -1.0, 1.0),
        np.clip(confidence, 0.0, 1.0),
        
        # Spatial features (7-9)
        np.clip(center_x, 0.0, 1.0),
        np.clip(center_y, 0.0, 1.0),
        np.clip(bbox_area, 0.0, 1.0),
        
        # Enhanced derived features (10-11)
        float(look_flag and gesture_flag),  # Combined suspicious behavior
        abs(center_x - 0.5) + abs(center_y - 0.5)  # Spatial center offset
    ], dtype=np.float32)
    
    return enhanced_vector

# Global enhanced classifier instance
_enhanced_lstm_classifier = None

def get_enhanced_lstm_classifier(model_path: str = "weights/enhanced_lstm_behavior.pth",
                               label_encoder_path: str = "weights/enhanced_label_encoder.pkl") -> EnhancedLSTMClassifier:
    """Get global enhanced LSTM classifier instance"""
    global _enhanced_lstm_classifier
    
    if _enhanced_lstm_classifier is None:
        _enhanced_lstm_classifier = EnhancedLSTMClassifier(model_path, label_encoder_path)
        if not _enhanced_lstm_classifier.load_model():
            logger.warning("Enhanced LSTM model failed to load, using default configuration")
    
    return _enhanced_lstm_classifier

if __name__ == "__main__":
    # Test enhanced classifier
    logger.info("Testing Enhanced LSTM Classifier...")
    
    classifier = get_enhanced_lstm_classifier()
    
    # Test with sample enhanced feature vector
    test_sequence = np.random.rand(10, 12).astype(np.float32)
    result = classifier.predict_enhanced(test_sequence)
    
    logger.info(f"Test result: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
    logger.info(f"Gesture detected: {result['gesture_detected']} (confidence: {result['gesture_confidence']:.3f})")
    logger.info(f"Looking detected: {result['looking_detected']} (confidence: {result['looking_confidence']:.3f})")
    logger.info("✅ Enhanced LSTM Classifier ready!")

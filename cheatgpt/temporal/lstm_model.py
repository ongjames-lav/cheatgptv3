"""
LSTM Model Loader for CheatGPT3 Engine Integration
=================================================

This module provides utilities to load and use the trained LSTM model
within the real-time engine for behavior classification.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class BehaviorLSTM(nn.Module):
    """LSTM network for temporal behavior pattern recognition."""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=5, 
                 sequence_length=20, dropout=0.3):
        super(BehaviorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through LSTM."""
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classify
        output = self.classifier(last_output)
        return output

class LSTMBehaviorClassifier:
    """Wrapper class for the trained LSTM behavior classifier."""
    
    def __init__(self, model_path: str = "weights/lstm_behavior.pth", 
                 label_encoder_path: str = "weights/label_encoder.pkl",
                 device: Optional[torch.device] = None):
        """
        Initialize the LSTM classifier.
        
        Args:
            model_path: Path to the trained model file
            label_encoder_path: Path to the label encoder file
            device: PyTorch device to use
        """
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.label_encoder = None
        self.class_labels = None
        self.is_loaded = False
        
        # Try to load the model
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained LSTM model and label encoder.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.warning(f"LSTM model not found at {self.model_path}")
                logger.info("Train the model first using: python train_lstm.py")
                return False
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model with saved configuration
            config = checkpoint['model_config']
            self.model = BehaviorLSTM(**config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"LSTM model loaded from {self.model_path}")
            logger.info(f"Model trained for {checkpoint['epoch']} epochs with {checkpoint['accuracy']:.2f}% accuracy")
            
            # Load label encoder if available
            if os.path.exists(self.label_encoder_path):
                with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                self.class_labels = list(self.label_encoder.classes_)  # Convert numpy array to list
                logger.info(f"Label encoder loaded: {self.class_labels}")
            else:
                # Default class labels if no encoder file
                self.class_labels = ['normal', 'leaning', 'looking_around', 'phone_use', 'cheating']
                logger.warning(f"Label encoder not found, using default labels: {self.class_labels}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, behavior_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Optimized prediction with enhanced preprocessing and error handling.
        
        Args:
            behavior_sequence: Array of shape (sequence_length, input_size) with features
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            return {
                'predicted_class': 0,
                'predicted_label': 'normal',
                'confidence': 0.5,
                'probabilities': None,
                'error': 'Model not loaded'
            }
        
        try:
            # OPTIMIZATION 1: Input validation and preprocessing
            if isinstance(behavior_sequence, list):
                if len(behavior_sequence) == 0:
                    raise ValueError("Empty behavior sequence provided")
                behavior_sequence = np.array(behavior_sequence, dtype=np.float32)
            elif not isinstance(behavior_sequence, np.ndarray):
                raise ValueError(f"Invalid input type: {type(behavior_sequence)}")
            elif behavior_sequence.size == 0:
                raise ValueError("Empty numpy array provided")
            elif behavior_sequence.dtype != np.float32:
                behavior_sequence = behavior_sequence.astype(np.float32)
            
            # Validate array shape
            if behavior_sequence.ndim == 0:
                raise ValueError("Invalid array: scalar provided")
            elif behavior_sequence.ndim == 1:
                if behavior_sequence.size == 0:
                    raise ValueError("Empty 1D array provided")
            elif behavior_sequence.ndim == 2:
                if behavior_sequence.shape[0] == 0 or behavior_sequence.shape[1] == 0:
                    raise ValueError("Empty 2D array provided")
            else:
                raise ValueError(f"Invalid array dimensions: {behavior_sequence.ndim}D")
            
            # OPTIMIZATION 2: Efficient shape handling
            original_shape = behavior_sequence.shape
            target_seq_len = self.model.sequence_length
            target_input_size = self.model.lstm.input_size
            
            # Handle sequence length
            if len(original_shape) == 1:
                # Single frame - create sequence by replication
                behavior_sequence = np.tile(behavior_sequence.reshape(1, -1), (target_seq_len, 1))
            elif original_shape[0] < target_seq_len:
                # Pad sequence using last frame replication (more stable than zeros)
                if behavior_sequence.size > 0 and original_shape[0] > 0:
                    last_frame = behavior_sequence[-1:] 
                else:
                    last_frame = np.zeros((1, original_shape[1]), dtype=np.float32)
                padding_needed = target_seq_len - original_shape[0]
                padding = np.tile(last_frame, (padding_needed, 1))
                behavior_sequence = np.vstack([behavior_sequence, padding])
            elif original_shape[0] > target_seq_len:
                # Use sliding window for temporal consistency
                behavior_sequence = behavior_sequence[-target_seq_len:]
            
            # Handle feature dimension
            current_features = behavior_sequence.shape[1]
            if current_features != target_input_size:
                if current_features < target_input_size:
                    # Pad features with zeros
                    padding = np.zeros((behavior_sequence.shape[0], target_input_size - current_features), dtype=np.float32)
                    behavior_sequence = np.hstack([behavior_sequence, padding])
                else:
                    # Truncate features to expected size
                    behavior_sequence = behavior_sequence[:, :target_input_size]
                    
                logger.debug(f"Feature dimension adjusted: {current_features} -> {target_input_size}")
            
            # OPTIMIZATION 3: Batch processing optimization
            # Add batch dimension efficiently
            input_tensor = torch.from_numpy(behavior_sequence).unsqueeze(0)
            
            # Move to device only once
            if input_tensor.device != self.device:
                input_tensor = input_tensor.to(self.device, non_blocking=True)
            
            # OPTIMIZATION 4: Inference optimization
            with torch.no_grad():
                # Set model to eval mode for consistent inference
                self.model.eval()
                
                # Forward pass
                outputs = self.model(input_tensor)
                
                # Compute probabilities using more numerically stable softmax
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_tensor = torch.argmax(probabilities, dim=1)
                predicted_class = int(predicted_class_tensor.item())  # Ensure it's a Python int
                confidence = float(probabilities[0][predicted_class].item())  # Ensure it's a Python float
                
                # OPTIMIZATION 5: Enhanced confidence calibration
                # Apply confidence calibration based on prediction certainty
                max_prob = float(confidence)
                second_max_prob = float(torch.topk(probabilities, 2, dim=1)[0][0][1].item())
                certainty_margin = max_prob - second_max_prob
                
                # Adjust confidence based on margin (more certain if margin is large)
                calibrated_confidence = max_prob * min(1.0, 0.5 + certainty_margin)
                
                # OPTIMIZATION 6: Label mapping with validation
                if self.class_labels and 0 <= predicted_class < len(self.class_labels):
                    predicted_label = self.class_labels[predicted_class]
                else:
                    predicted_label = f"class_{predicted_class}"
                    logger.warning(f"Predicted class {predicted_class} not in class_labels")
                
                # OPTIMIZATION 7: Enhanced result compilation
                result = {
                    'predicted_class': predicted_class,
                    'predicted_label': predicted_label,
                    'confidence': calibrated_confidence,
                    'raw_confidence': confidence,
                    'certainty_margin': certainty_margin,
                    'probabilities': probabilities[0].cpu().numpy().tolist(),
                    'class_labels': self.class_labels,
                    'sequence_length_used': target_seq_len,
                    'input_features_used': target_input_size,
                    'error': None
                }
                
                return result
            
        except Exception as e:
            import traceback
            logger.error(f"LSTM prediction error: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'predicted_class': 0,
                'predicted_label': 'normal',
                'confidence': 0.5,
                'raw_confidence': 0.5,
                'certainty_margin': 0.0,
                'probabilities': None,
                'class_labels': self.class_labels,
                'sequence_length_used': 0,
                'input_features_used': 0,
                'error': str(e)
            }
    
    def predict_batch(self, behavior_sequences: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Predict behaviors for a batch of sequences.
        
        Args:
            behavior_sequences: List of behavior sequences
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(seq) for seq in behavior_sequences]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'input_size': self.model.lstm.input_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'num_classes': self.model.num_classes,
            'sequence_length': self.model.sequence_length,
            'class_labels': self.class_labels,
            'is_loaded': self.is_loaded
        }

# Global instance for easy access
_global_classifier = None

def get_lstm_classifier(model_path: str = "weights/lstm_behavior.pth",
                       label_encoder_path: str = "weights/label_encoder.pkl",
                       device: Optional[torch.device] = None) -> LSTMBehaviorClassifier:
    """
    Get or create the global LSTM classifier instance.
    
    Args:
        model_path: Path to the trained model file
        label_encoder_path: Path to the label encoder file
        device: PyTorch device to use
        
    Returns:
        LSTMBehaviorClassifier instance
    """
    global _global_classifier
    
    if _global_classifier is None:
        _global_classifier = LSTMBehaviorClassifier(model_path, label_encoder_path, device)
    
    return _global_classifier

def reset_classifier():
    """Reset the global classifier instance."""
    global _global_classifier
    _global_classifier = None
